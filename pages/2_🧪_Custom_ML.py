import json

import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score, get_scorer, precision_score, recall_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score

from helpers.config import PIPELINES_DIR
from helpers.data_utils import read_sql_data, read_uploaded_data, split_context
from helpers.logging_utils import setup_logger
from helpers.ml_utils import (apply_smote_if_needed, available_scorers, build_feature_pipeline, cv_object,
                              evaluate_metrics, feature_select, get_feature_names, list_saved_pipelines,
                              load_pipeline, model_candidates, model_doc, objective_factory, param_grids,
                              save_model_bundle, save_pipeline, tuning_trials)
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


def run_training(
    ctx,
    metric: str,
    folds: int,
    tuning_level: str,
    imbalance: str,
    model_names: list[str] | None = None,
    use_stacking: bool = False,
    stacking_base_models: list[str] | None = None,
):
    x_train = st.session_state.selected_features["x_train"]
    y_train = ctx.y_train
    x_train, y_train = apply_smote_if_needed(x_train, y_train, ctx.task_type == "classification" and imbalance == "smote")
    cv = cv_object(ctx.task_type, folds)

    all_models = model_candidates(ctx.task_type)
    if model_names is not None:
        all_models = {k: v for k, v in all_models.items() if k in model_names}

    rows = []
    failed = []
    fitted_for_stacking = []
    trained_models = {}
    best_score = -np.inf
    best_model = st.session_state.best_model
    best_name = st.session_state.best_model_name
    scorer = get_scorer(metric)

    progress = st.progress(0)
    status = st.empty()

    total = max(1, len(all_models) + (1 if use_stacking else 0))
    current_stage = 0

    for name, base in all_models.items():
        current_stage += 1
        try:
            status.info(f"Training stage {current_stage}/{total}: {name}")
            logger.info("Training model stage %s/%s: %s", current_stage, total, name)

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
            trained_models[name] = model
            fitted_for_stacking.append((name, model))
            in_metrics = evaluate_metrics(model, ctx.task_type, x_train, y_train)
            out_metrics = evaluate_metrics(model, ctx.task_type, st.session_state.selected_features["x_test"], ctx.y_test)
            out_val = float(scorer(model, st.session_state.selected_features["x_test"], ctx.y_test))
            boot_vals = []
            n_obs = len(ctx.y_test)
            for _ in range(30):
                bs_idx = np.random.randint(0, n_obs, n_obs)
                boot_vals.append(float(scorer(model, st.session_state.selected_features["x_test"][bs_idx], ctx.y_test.iloc[bs_idx])))
            boot_mean, boot_std = float(np.mean(boot_vals)), float(np.std(boot_vals))

            rows.append(
                {
                    "model": name,
                    "metric_in_sample_mean_cv": float(cv_scores.mean()),
                    "metric_in_sample_std_cv": float(cv_scores.std()),
                    "metric_in_sample_mean_bootstrap": boot_mean,
                    "metric_in_sample_std_bootstrap": boot_std,
                    "metric_out_of_sample": out_val,
                    "all_in_sample_metrics": json.dumps(in_metrics),
                    "all_out_sample_metrics": json.dumps(out_metrics),
                    "best_params": json.dumps(study.best_params),
                }
            )

            if cv_scores.mean() > best_score:
                best_score = float(cv_scores.mean())
                best_model = model
                best_name = name

        except Exception as exc:
            err_text = str(exc)
            failed.append({"model": name, "error": err_text, "solution": suggest_solution(err_text)})
            logger.exception("Model %s failed: %s", name, err_text)

        progress.progress(min(1.0, current_stage / total))

    if use_stacking:
        current_stage += 1
        try:
            status.info(f"Training stage {current_stage}/{total}: stacking_ensemble")
            stack_names = stacking_base_models or [n for n, _ in fitted_for_stacking[:3]]
            estimators = [(n, m) for n, m in fitted_for_stacking if n in stack_names]
            if len(estimators) >= 2:
                final_est = LogisticRegression(max_iter=3000) if ctx.task_type == "classification" else LinearRegression()
                stack_model = StackingClassifier(estimators=estimators, final_estimator=final_est, cv=3) if ctx.task_type == "classification" else StackingRegressor(estimators=estimators, final_estimator=final_est, cv=3)
                cv_scores = cross_val_score(stack_model, x_train, y_train, cv=cv, scoring=metric)
                stack_model.fit(x_train, y_train)
                in_metrics = evaluate_metrics(stack_model, ctx.task_type, x_train, y_train)
                out_metrics = evaluate_metrics(stack_model, ctx.task_type, st.session_state.selected_features["x_test"], ctx.y_test)
                out_val = float(scorer(stack_model, st.session_state.selected_features["x_test"], ctx.y_test))
                boot_vals = []
                n_obs = len(ctx.y_test)
                for _ in range(30):
                    bs_idx = np.random.randint(0, n_obs, n_obs)
                    boot_vals.append(float(scorer(stack_model, st.session_state.selected_features["x_test"][bs_idx], ctx.y_test.iloc[bs_idx])))
                boot_mean, boot_std = float(np.mean(boot_vals)), float(np.std(boot_vals))
                rows.append(
                    {
                        "model": "stacking_ensemble",
                        "metric_in_sample_mean_cv": float(cv_scores.mean()),
                        "metric_in_sample_std_cv": float(cv_scores.std()),
                        "metric_in_sample_mean_bootstrap": boot_mean,
                        "metric_in_sample_std_bootstrap": boot_std,
                        "metric_out_of_sample": out_val,
                        "all_in_sample_metrics": json.dumps(in_metrics),
                        "all_out_sample_metrics": json.dumps(out_metrics),
                        "best_params": json.dumps({"estimators": stack_names}),
                    }
                )
                trained_models["stacking_ensemble"] = stack_model
                if cv_scores.mean() > best_score:
                    best_score = float(cv_scores.mean())
                    best_model = stack_model
                    best_name = "stacking_ensemble"
        except Exception as exc:
            err_text = str(exc)
            failed.append({"model": "stacking_ensemble", "error": err_text, "solution": suggest_solution(err_text)})

        progress.progress(min(1.0, current_stage / total))

    status.success("Training completed")
    return rows, failed, best_model, best_name, trained_models


@st.cache_data(show_spinner=False)
def _corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


tabs = st.tabs(["1) Import & Split", "2) EDA", "3) Feature Engineering", "4) Feature Selection", "5) Modeling", "6) Results", "7) XAI"])

with tabs[0]:
    source = st.radio("Data source", ["Upload file", "SQL query"], horizontal=True, key="custom_source")
    df = None
    if source == "Upload file":
        file_type = st.selectbox("File type", ["csv", "json", "xlsx"], key="custom_file_type")
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
        target = st.selectbox("Target", st.session_state.raw_df.columns.tolist(), key="custom_target")
        task_type = st.selectbox("Task type", ["classification", "regression"], key="custom_task_type")
        st.caption("Model documentation")
        st.json(model_doc(task_type))
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="custom_test_size")
        st.caption("Random state is fixed at 42 for reproducibility.")
        if st.button("Create split"):
            st.session_state.ctx = split_context(st.session_state.raw_df, target, task_type, test_size, 42)

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
        num_fill = st.selectbox("Numeric NA strategy", ["mean", "median"], key="custom_num_fill")
        cat_fill = st.selectbox("Categorical NA strategy", ["most_frequent", "constant"], key="custom_cat_fill")
        scaler = st.selectbox("Scaling", ["standard", "robust"], key="custom_scaler")
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
    method = st.selectbox("Selection method", ["manual", "kbest", "mutual_info", "rfe"], key="custom_selection_method")
    total_feats = st.session_state.x_train_ready.shape[1]
    k = st.slider("Top-k features", 5, max(5, total_feats), min(25, total_feats), key="custom_selection_k")
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
        st.session_state.selection_method = method
        st.session_state.selection_top_k = int(k)

if st.session_state.selected_features is None:
    st.stop()

with tabs[4]:
    ctx = st.session_state.ctx
    scorers = available_scorers(ctx.task_type)
    default_metric = "f1_weighted" if ctx.task_type == "classification" else "neg_root_mean_squared_error"
    metric = st.selectbox("Optimization metric", scorers, index=scorers.index(default_metric) if default_metric in scorers else 0, key="custom_metric")
    folds = st.slider("CV folds", 3, 10, 5, key="custom_folds")
    tuning_level = st.select_slider("Optuna tuning level", options=["basic", "light", "medium", "heavy", "extreme"], key="custom_tuning_level")
    imbalance = st.selectbox("Imbalance strategy", ["none", "class_weight", "smote"], key="custom_imbalance") if ctx.task_type == "classification" else "none"
    available_algorithms = list(model_candidates(ctx.task_type).keys())
    selected_algorithms = st.multiselect("Algorithms to run", available_algorithms, default=available_algorithms, key="custom_selected_algorithms")
    use_stacking = st.checkbox("Enable stacking ensemble", key="custom_use_stacking")
    selected_stacking_models = st.multiselect("Stacking base models", available_algorithms, default=available_algorithms[:3], key="custom_stacking_models") if use_stacking else []


    st.subheader("Training configuration summary")
    training_summary = {
        "mode": "Custom ML",
        "task": ctx.task_type,
        "target": ctx.target,
        "train_rows": int(len(ctx.y_train)),
        "test_rows": int(len(ctx.y_test)),
        "n_raw_features": int(len(ctx.feature_cols)),
        "n_engineered_features": int(st.session_state.selected_features["x_train"].shape[1]),
        "feature_selection_method": st.session_state.get("selection_method", "not-set"),
        "feature_selection_top_k": st.session_state.get("selection_top_k", "not-set"),
        "optimization_metric": metric,
        "cv_folds": int(folds),
        "optuna_tuning_level": tuning_level,
        "imbalance_strategy": imbalance,
        "algorithms_to_run": selected_algorithms,
        "stacking_enabled": bool(use_stacking),
        "stacking_base_models": selected_stacking_models,
    }
    st.json(training_summary)

    if st.button("Run modeling"):
        if not selected_algorithms:
            st.error("Please select at least one algorithm to run.")
            st.stop()
        rows, failed, best_model, best_name, trained_models = run_training(ctx, metric, folds, tuning_level, imbalance, model_names=selected_algorithms, use_stacking=use_stacking, stacking_base_models=selected_stacking_models)
        models_summary = pd.DataFrame(rows)
        if models_summary.empty:
            st.session_state.models_summary = None
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_name
            st.session_state.failed_runs = failed
            st.session_state.trained_models = trained_models
            st.error("No models completed successfully. Review diagnostics and retry with different settings.")
            if failed:
                st.dataframe(pd.DataFrame(failed))
            st.stop()

        sort_col = "metric_in_sample_mean_cv" if "metric_in_sample_mean_cv" in models_summary.columns else "metric_out_of_sample"
        st.session_state.models_summary = models_summary.sort_values(sort_col, ascending=False)
        st.session_state.best_model = best_model
        st.session_state.best_model_name = best_name
        st.session_state.failed_runs = failed
        st.session_state.trained_models = trained_models

        if failed:
            st.error("Some models failed during training. See diagnostics below.")
            st.dataframe(pd.DataFrame(failed))
        else:
            st.success("All models completed successfully.")

if st.session_state.models_summary is None:
    st.stop()

with tabs[5]:
    st.dataframe(st.session_state.models_summary)
    if "metric_in_sample_mean_cv" in st.session_state.models_summary.columns:
        st.plotly_chart(px.bar(st.session_state.models_summary, x="model", y="metric_in_sample_mean_cv", title="CV Mean"), use_container_width=True)
    if {"metric_in_sample_mean_cv", "metric_out_of_sample"}.issubset(st.session_state.models_summary.columns):
        st.plotly_chart(px.scatter(st.session_state.models_summary, x="metric_in_sample_mean_cv", y="metric_out_of_sample", text="model", title="In vs Out"), use_container_width=True)
    if "metric_in_sample_std_cv" in st.session_state.models_summary.columns:
        st.plotly_chart(px.bar(st.session_state.models_summary, x="model", y="metric_in_sample_std_cv", title="CV Std"), use_container_width=True)

    if "all_out_sample_metrics" in st.session_state.models_summary.columns:
        parsed_metrics = st.session_state.models_summary["all_out_sample_metrics"].apply(lambda x: json.loads(x))
        all_metric_names = sorted({k for d in parsed_metrics for k, v in d.items() if isinstance(v, (int, float))})
        if all_metric_names:
            selected_result_metric = st.selectbox("Second results table metric", all_metric_names)
            metric_rows = []
            for _, r in st.session_state.models_summary.iterrows():
                m = json.loads(r["all_out_sample_metrics"]).get(selected_result_metric)
                metric_rows.append({"model": r["model"], selected_result_metric: m})
            metric_df = pd.DataFrame(metric_rows).sort_values(selected_result_metric, ascending=False)
            st.dataframe(metric_df)

    details_model = st.selectbox("Model metric details", st.session_state.models_summary["model"].tolist())
    row = st.session_state.models_summary[st.session_state.models_summary["model"] == details_model].iloc[0]
    if "all_in_sample_metrics" in row and pd.notna(row["all_in_sample_metrics"]):
        st.write("In-sample metrics")
        st.json(json.loads(row["all_in_sample_metrics"]))
    if "all_out_sample_metrics" in row and pd.notna(row["all_out_sample_metrics"]):
        st.write("Out-of-sample metrics")
        st.json(json.loads(row["all_out_sample_metrics"]))

    if st.session_state.ctx.task_type == "classification":
        import matplotlib.pyplot as plt

        y_test = st.session_state.ctx.y_test
        x_test = st.session_state.selected_features["x_test"]
        y_pred = st.session_state.best_model.predict(x_test)

        denominator = st.selectbox("Confusion matrix denominator", ["none", "true", "pred", "all"], index=0)
        denom_arg = None if denominator == "none" else denominator
        fig_in, ax_in = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=denom_arg, ax=ax_in)
        st.pyplot(fig_in)

        labels = sorted(st.session_state.ctx.y_test.unique().tolist())
        label_options = sorted(set(labels + [0, 1]))
        selected_label = st.selectbox("Label-specific metrics", label_options, help="Choose class label for precision/recall/F1.")

        y_train_pred = st.session_state.best_model.predict(st.session_state.selected_features["x_train"])
        y_test_pred = st.session_state.best_model.predict(x_test)
        binary_case = len(labels) == 2 and selected_label in labels
        in_label_metrics = {
            "precision_label": precision_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else precision_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
            "recall_label": recall_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else recall_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
            "f1_label": f1_score(st.session_state.ctx.y_train, y_train_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else f1_score(st.session_state.ctx.y_train, y_train_pred, labels=[selected_label], average="macro", zero_division=0),
        }
        out_label_metrics = {
            "precision_label": precision_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else precision_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
            "recall_label": recall_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else recall_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
            "f1_label": f1_score(y_test, y_test_pred, pos_label=selected_label, average="binary", zero_division=0) if binary_case else f1_score(y_test, y_test_pred, labels=[selected_label], average="macro", zero_division=0),
        }
        st.write(f"In-sample metrics for label: {selected_label}")
        st.json(in_label_metrics)
        st.write(f"Out-of-sample metrics for label: {selected_label}")
        st.json(out_label_metrics)

        if hasattr(st.session_state.best_model, "predict_proba") and binary_case:
            probs = st.session_state.best_model.predict_proba(x_test)[:, 1]
            threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
            th_preds = (probs >= threshold).astype(int)
            st.write("Threshold-based metrics")
            st.json(
                {
                    "precision": precision_score(y_test, th_preds, zero_division=0),
                    "recall": recall_score(y_test, th_preds, zero_division=0),
                    "f1": f1_score(y_test, th_preds, zero_division=0),
                }
            )
            thresholds = np.linspace(0.01, 0.99, 99)
            curve = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "precision": [precision_score(y_test, (probs >= t).astype(int), zero_division=0) for t in thresholds],
                    "recall": [recall_score(y_test, (probs >= t).astype(int), zero_division=0) for t in thresholds],
                    "f1": [f1_score(y_test, (probs >= t).astype(int), zero_division=0) for t in thresholds],
                }
            )
            st.plotly_chart(px.line(curve, x="threshold", y=["precision", "recall", "f1"], title="Threshold tuning curves"), use_container_width=True)
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            st.plotly_chart(px.line(roc_df, x="fpr", y="tpr", title="ROC Curve"), use_container_width=True)
            prob_df = pd.DataFrame({"probability": probs})
            st.plotly_chart(px.histogram(prob_df, x="probability", nbins=30, title="Predicted probability distribution"), use_container_width=True)

    if st.session_state.ctx.task_type == "regression":
        y_test = st.session_state.ctx.y_test
        x_test = st.session_state.selected_features["x_test"]
        y_pred = st.session_state.best_model.predict(x_test)
        reg_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
        reg_df["residual"] = reg_df["actual"] - reg_df["predicted"]
        st.plotly_chart(px.scatter(reg_df, x="actual", y="predicted", title="Predicted vs Actual"), use_container_width=True)
        st.plotly_chart(px.histogram(reg_df, x="residual", nbins=40, title="Residual Distribution"), use_container_width=True)
        st.plotly_chart(px.scatter(reg_df.reset_index(), x=reg_df.index, y="residual", title="Residuals by Observation"), use_container_width=True)

    models_to_save = list(st.session_state.get("trained_models", {}).keys())
    if models_to_save:
        selected_model_to_save = st.selectbox("Choose model to save (end-to-end bundle)", models_to_save)
        model_name = st.text_input("Save model bundle as", value=f"{selected_model_to_save}_bundle.joblib")
        if st.button("Save selected model bundle"):
            model_obj = st.session_state.trained_models[selected_model_to_save]
            st.success(
                f"Saved bundle to {save_model_bundle(model_obj, st.session_state.feature_pipeline, st.session_state.selected_features['selector'], st.session_state.ctx.task_type, get_feature_names(st.session_state.feature_pipeline), model_name)}"
            )

with tabs[6]:
    feature_names = get_feature_names(st.session_state.feature_pipeline)
    selector = st.session_state.selected_features["selector"]
    if selector is not None and hasattr(selector, "get_support"):
        mask = selector.get_support()
        feature_names = [f for f, keep in zip(feature_names, mask) if keep]

    x_perm = st.session_state.selected_features["x_test"]
    if sparse.issparse(x_perm):
        x_perm = x_perm.toarray()
    importance = permutation_importance(st.session_state.best_model, x_perm, st.session_state.ctx.y_test, n_repeats=5, random_state=42)
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
    st.plotly_chart(px.bar(imp_df.head(25), x="feature", y="importance", title="Permutation Importance"), use_container_width=True)

    if hasattr(st.session_state.best_model, "feature_importances_"):
        native = pd.DataFrame({"feature": feature_names, "importance": st.session_state.best_model.feature_importances_}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(native.head(25), x="feature", y="importance", title="Model Feature Importance"), use_container_width=True)

    if not imp_df.empty:
        import matplotlib.pyplot as plt

        top_feature = imp_df.iloc[0]["feature"]
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(st.session_state.best_model, x_perm, [feature_names.index(top_feature)], ax=ax)
        st.pyplot(fig)
