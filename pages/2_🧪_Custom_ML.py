import json

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

from helpers.config import PIPELINES_DIR
from helpers.data_utils import read_sql_data, read_uploaded_data, split_context
from helpers.logging_utils import setup_logger
from helpers.ml_utils import (apply_smote_if_needed, available_scorers, bootstrap_metric, build_feature_pipeline,
                              cv_object, evaluate_metrics, feature_select, get_feature_names, list_saved_pipelines,
                              load_pipeline, model_candidates, objective_factory, save_model_bundle,
                              save_pipeline, summary_metric_name, tuning_trials)
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()
logger = setup_logger(verbose=True)


def suggest_solution(error_text: str) -> str:
    msg = error_text.lower()
    if "could not convert" in msg or "dtype" in msg:
        return "Check column types and ensure categorical values are encoded consistently."
    if "nan" in msg or "infinity" in msg:
        return "Review missing values and apply robust imputation before training."
    if "memory" in msg:
        return "Reduce Optuna trials, use fewer models, or sample fewer rows/features."
    if "class" in msg and "only one" in msg:
        return "Target has only one class in train split; use stratified split or more balanced data."
    return "Try fewer folds/trials, verify target/feature selections, and rerun failed models."


def run_training(ctx, metric: str, folds: int, tuning_level: str, imbalance: str, model_names: list[str] | None = None):
    x_train = st.session_state.selected_features["x_train"]
    y_train = ctx.y_train
    x_train, y_train = apply_smote_if_needed(x_train, y_train, ctx.task_type == "classification" and imbalance == "smote")
    cv = cv_object(ctx.task_type, folds)

    all_models = model_candidates(ctx.task_type)
    if model_names is not None:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    rows = []
    failed = []
    best_score = -np.inf
    best_model = st.session_state.best_model
    best_name = st.session_state.best_model_name
    summary_metric = summary_metric_name(ctx.task_type)

    progress = st.progress(0)
    status = st.empty()

    total = max(1, len(all_models))
    for idx, (name, base) in enumerate(all_models.items(), start=1):
        try:
            status.info(f"Training stage {idx}/{total}: {name}")
            logger.info("Training model stage %s/%s: %s", idx, total, name)

            model = clone(base)
            if ctx.task_type == "classification" and imbalance == "class_weight" and hasattr(model, "class_weight"):
                model.set_params(class_weight="balanced")

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective_factory(name, model, x_train, y_train, cv, metric),
                n_trials=tuning_trials(tuning_level),
                show_progress_bar=False,
            )
            if study.best_params:
                model.set_params(**study.best_params)

            cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=metric)
            model.fit(x_train, y_train)
            in_metrics = evaluate_metrics(model, ctx.task_type, x_train, y_train)
            out_metrics = evaluate_metrics(model, ctx.task_type, st.session_state.selected_features["x_test"], ctx.y_test)
            boot_mean, boot_std = bootstrap_metric(model, ctx.task_type, st.session_state.selected_features["x_test"], ctx.y_test, summary_metric)
            out_val = out_metrics.get(summary_metric)

            rows.append(
                {
                    "model": name,
                    "metric_in_sample_mean_cv": float(cv_scores.mean()),
                    "metric_in_sample_std_cv": float(cv_scores.std()),
                    "metric_in_sample_mean_bootstrap": boot_mean,
                    "metric_in_sample_std_bootstrap": boot_std,
                    "metric_out_of_sample": float(out_val) if isinstance(out_val, (float, int, np.floating)) else np.nan,
                    "all_in_sample_metrics": json.dumps(in_metrics),
                    "all_out_sample_metrics": json.dumps(out_metrics),
                    "best_params": json.dumps(study.best_params),
                }
            )

            if cv_scores.mean() > best_score:
                best_score = float(cv_scores.mean())
                best_model = model
                best_name = name

            logger.info("Model %s completed successfully", name)
        except Exception as exc:
            err_text = str(exc)
            solution = suggest_solution(err_text)
            logger.exception("Model %s failed: %s", name, err_text)
            failed.append({"model": name, "error": err_text, "solution": solution})

        progress.progress(min(1.0, idx / total))

    status.success("Training completed")
    return rows, failed, best_model, best_name


@st.cache_data(show_spinner=False)
def _corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


st.header("AI λab — Custom ML")
tabs = st.tabs(["1) Import & Split", "2) EDA", "3) Feature Engineering", "4) Feature Selection", "5) Modeling", "6) Results", "7) XAI"])

with tabs[0]:
    source = st.radio("Data source", ["Upload file", "SQL query"], horizontal=True)
    df = None
    if source == "Upload file":
        file_type = st.selectbox("File type", ["csv", "json", "xlsx"])
        uploaded = st.file_uploader("Upload data", type=["csv", "json", "xlsx"])
        if uploaded is not None:
            df = read_uploaded_data(uploaded, file_type)
    else:
        conn = st.text_input("SQLAlchemy URI", placeholder="sqlite:///my.db")
        query = st.text_area("SQL query", value="SELECT * FROM your_table")
        if st.button("Run SQL"):
            try:
                df = read_sql_data(conn, query)
            except Exception as exc:
                st.error(f"SQL error: {exc}")

    if df is not None:
        st.session_state.raw_df = df

    if st.session_state.raw_df is not None:
        st.dataframe(st.session_state.raw_df.head(100))
        target = st.selectbox("Target", st.session_state.raw_df.columns.tolist())
        task_type = st.selectbox("Task type", ["classification", "regression"])
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random state", 0, 9999, 42)
        if st.button("Create split"):
            st.session_state.ctx = split_context(st.session_state.raw_df, target, task_type, test_size, int(random_state))

if st.session_state.ctx is None:
    st.stop()

with tabs[1]:
    st.dataframe(st.session_state.ctx.train_df.describe(include="all").transpose())
    na_pct = st.session_state.ctx.train_df.isna().mean().mul(100).sort_values(ascending=False).reset_index()
    na_pct.columns = ["feature", "na_pct"]
    st.plotly_chart(px.bar(na_pct, x="feature", y="na_pct", title="Missing values (%)"), use_container_width=True)

    numeric_cols = st.session_state.ctx.train_df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        corr = _corr_matrix(st.session_state.ctx.train_df[numeric_cols])
        st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation matrix"), use_container_width=True)

with tabs[2]:
    st.subheader("Feature pipeline")
    existing_pipelines = list_saved_pipelines()
    mode = st.radio("Pipeline option", ["Create new pipeline", "Load existing pipeline"], horizontal=True)
    if mode == "Create new pipeline":
        num_fill = st.selectbox("Numeric NA strategy", ["mean", "median"])
        cat_fill = st.selectbox("Categorical NA strategy", ["most_frequent", "constant"])
        scaler = st.selectbox("Scaling", ["standard", "robust"])
        if st.button("Build transformations"):
            ctx = st.session_state.ctx
            pipeline = build_feature_pipeline(ctx.train_df[ctx.feature_cols], scaler, cat_fill, num_fill)
            x_train = pipeline.fit_transform(ctx.train_df[ctx.feature_cols], ctx.y_train)
            x_test = pipeline.transform(ctx.test_df[ctx.feature_cols])
            st.session_state.feature_pipeline = pipeline
            st.session_state.x_train_ready = x_train
            st.session_state.x_test_ready = x_test
            st.success("Pipeline trained on train and applied on test")
    else:
        if not existing_pipelines:
            st.info(f"No pipeline found in {PIPELINES_DIR}")
        else:
            selected_pipeline = st.selectbox("Saved pipeline", existing_pipelines)
            if st.button("Load pipeline"):
                ctx = st.session_state.ctx
                pipeline = load_pipeline(selected_pipeline)
                x_train = pipeline.transform(ctx.train_df[ctx.feature_cols])
                x_test = pipeline.transform(ctx.test_df[ctx.feature_cols])
                st.session_state.feature_pipeline = pipeline
                st.session_state.x_train_ready = x_train
                st.session_state.x_test_ready = x_test
                st.success(f"Loaded pipeline: {selected_pipeline}")

if st.session_state.feature_pipeline is None:
    st.stop()

with tabs[3]:
    method = st.selectbox("Selection method", ["manual", "kbest", "mutual_info", "rfe"])
    total_feats = st.session_state.x_train_ready.shape[1]
    k = st.slider("Top-k features", 5, max(5, total_feats), min(25, total_feats))
    if st.button("Apply feature selection"):
        x_train_selected, selector = feature_select(
            st.session_state.x_train_ready,
            st.session_state.ctx.y_train,
            st.session_state.ctx.task_type,
            method,
            k,
        )
        x_test_selected = selector.transform(st.session_state.x_test_ready) if selector is not None else st.session_state.x_test_ready
        st.session_state.selected_features = {"selector": selector, "x_train": x_train_selected, "x_test": x_test_selected}

if st.session_state.selected_features is None:
    st.stop()

with tabs[4]:
    ctx = st.session_state.ctx
    scorers = available_scorers(ctx.task_type)
    default_metric = "f1_weighted" if ctx.task_type == "classification" else "neg_root_mean_squared_error"
    metric = st.selectbox("Optimization metric", scorers, index=scorers.index(default_metric) if default_metric in scorers else 0)
    folds = st.slider("CV folds", 3, 10, 5)
    tuning_level = st.select_slider("Optuna tuning level", options=["basic", "light", "medium", "heavy", "extreme"])
    imbalance = st.selectbox("Imbalance strategy", ["none", "class_weight", "smote"]) if ctx.task_type == "classification" else "none"

    if st.button("Run modeling"):
        rows, failed, best_model, best_name = run_training(ctx, metric, folds, tuning_level, imbalance)
        st.session_state.models_summary = pd.DataFrame(rows).sort_values("metric_in_sample_mean_cv", ascending=False)
        st.session_state.best_model = best_model
        st.session_state.best_model_name = best_name
        st.session_state.failed_runs = failed

        if failed:
            st.error("Some models failed during training. See diagnostics below.")
            st.dataframe(pd.DataFrame(failed))
        else:
            st.success("All models completed successfully.")

    if st.session_state.get("failed_runs"):
        st.warning("You have failed models from the previous run.")
        if st.button("Rerun failed models only"):
            failed_models = [item["model"] for item in st.session_state["failed_runs"]]
            rows, failed, best_model, best_name = run_training(ctx, metric, folds, tuning_level, imbalance, failed_models)
            existing = st.session_state.models_summary if st.session_state.models_summary is not None else pd.DataFrame()
            rerun_df = pd.DataFrame(rows)
            combined = pd.concat([existing[~existing["model"].isin(failed_models)], rerun_df], ignore_index=True)
            st.session_state.models_summary = combined.sort_values("metric_in_sample_mean_cv", ascending=False)
            st.session_state.best_model = best_model if best_model is not None else st.session_state.best_model
            st.session_state.best_model_name = best_name if best_name is not None else st.session_state.best_model_name
            st.session_state.failed_runs = failed
            if failed:
                st.error("Some failed models still require attention.")
                st.dataframe(pd.DataFrame(failed))
            else:
                st.success("Failed models rerun succeeded.")

if st.session_state.models_summary is None:
    st.stop()

with tabs[5]:
    st.dataframe(st.session_state.models_summary)
    st.plotly_chart(px.bar(st.session_state.models_summary, x="model", y="metric_in_sample_mean_cv", title="CV Mean"), use_container_width=True)
    st.plotly_chart(px.scatter(st.session_state.models_summary, x="metric_in_sample_mean_cv", y="metric_out_of_sample", text="model", title="In vs Out"), use_container_width=True)
    st.plotly_chart(px.bar(st.session_state.models_summary, x="model", y="metric_in_sample_std_cv", title="CV Std"), use_container_width=True)

    details_model = st.selectbox("Model metric details", st.session_state.models_summary["model"].tolist())
    row = st.session_state.models_summary[st.session_state.models_summary["model"] == details_model].iloc[0]
    st.write("In-sample metrics")
    st.json(json.loads(row["all_in_sample_metrics"]))
    st.write("Out-of-sample metrics")
    st.json(json.loads(row["all_out_sample_metrics"]))

    if st.session_state.ctx.task_type == "classification":
        import matplotlib.pyplot as plt

        y_test = st.session_state.ctx.y_test
        x_test = st.session_state.selected_features["x_test"]
        y_pred = st.session_state.best_model.predict(x_test)
        fig_in, ax_in = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=None, ax=ax_in)
        st.pyplot(fig_in)
        fig_pct, ax_pct = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true", ax=ax_pct)
        st.pyplot(fig_pct)

        labels = sorted(st.session_state.ctx.y_test.unique().tolist())
        selected_label = st.selectbox("Label-specific metrics", labels, help="Choose the class label to inspect precision/recall/F1 for this specific class.")

        y_train_pred = st.session_state.best_model.predict(st.session_state.selected_features["x_train"])
        y_test_pred = st.session_state.best_model.predict(x_test)
        in_label_metrics = {
            "precision_label": precision_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else precision_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
            "recall_label": recall_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else recall_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
            "f1_label": f1_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else f1_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
        }
        out_label_metrics = {
            "precision_label": precision_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else precision_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
            "recall_label": recall_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else recall_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
            "f1_label": f1_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if len(labels)==2 else f1_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
        }
        st.write(f"In-sample metrics for label: {selected_label}")
        st.json(in_label_metrics)
        st.write(f"Out-of-sample metrics for label: {selected_label}")
        st.json(out_label_metrics)

    pipeline_name = st.text_input("Save pipeline as", value="feature_pipeline.joblib")
    if st.button("Save pipeline"):
        st.success(f"Saved pipeline to {save_pipeline(st.session_state.feature_pipeline, pipeline_name)}")

    model_name = st.text_input("Save model bundle as", value=f"{st.session_state.best_model_name}_bundle.joblib")
    if st.button("Save model bundle"):
        st.success(
            f"Saved bundle to {save_model_bundle(st.session_state.best_model, st.session_state.feature_pipeline, st.session_state.selected_features['selector'], st.session_state.ctx.task_type, get_feature_names(st.session_state.feature_pipeline), model_name)}"
        )

with tabs[6]:
    feature_names = get_feature_names(st.session_state.feature_pipeline)
    selector = st.session_state.selected_features["selector"]
    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        feature_names = [f for f, keep in zip(feature_names, mask) if keep]

    importance = permutation_importance(st.session_state.best_model, st.session_state.selected_features["x_test"], st.session_state.ctx.y_test, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    st.plotly_chart(px.bar(imp_df.head(25), x="feature", y="importance", title="Permutation Importance"), use_container_width=True)

    if hasattr(st.session_state.best_model, "feature_importances_"):
        native = pd.DataFrame({"feature": feature_names, "importance": st.session_state.best_model.feature_importances_}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(native.head(25), x="feature", y="importance", title="Model Feature Importance"), use_container_width=True)

    if not imp_df.empty:
        import matplotlib.pyplot as plt

        top_feature = imp_df.iloc[0]["feature"]
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(st.session_state.best_model, st.session_state.selected_features["x_test"], [feature_names.index(top_feature)], ax=ax)
        st.pyplot(fig)
