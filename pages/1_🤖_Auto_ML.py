import importlib.util

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import sparse
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, get_scorer
from sklearn.model_selection import cross_val_score

from helpers.data_utils import read_uploaded_data, split_context
from helpers.ml_utils import build_feature_pipeline, evaluate_metrics, get_feature_names, model_candidates
from helpers.state import initialize_session_state
from helpers.style import set_app_style

HAS_SHAP = importlib.util.find_spec("shap") is not None
HAS_LIME = importlib.util.find_spec("lime") is not None
if HAS_SHAP:
    import shap
if HAS_LIME:
    from lime.lime_tabular import LimeTabularExplainer

initialize_session_state()
set_app_style()

for key in ["automl_summary", "automl_model", "automl_pipeline", "automl_ctx", "automl_x_train", "automl_x_test", "automl_boot", "automl_metric_table", "automl_feature_names"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.subheader("Auto ML")
tab_labels = ["1) Import & Setup", "2) Modeling", "3) Results", "4) XAI"]
if "automl_tab_selected" not in st.session_state:
    st.session_state.automl_tab_selected = tab_labels[0]
tabs = st.tabs(tab_labels, default=st.session_state.automl_tab_selected, key="automl_tab_selector")

with tabs[0]:
    uploaded = st.file_uploader("Upload csv/json/xlsx", type=["csv", "json", "xlsx"], key="automl_upload")
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1]
        df = read_uploaded_data(uploaded, ext)
        c1, c2 = st.columns(2)
        with c1:
            task = st.selectbox("Task", ["classification", "regression"], key="automl_task")
        with c2:
            target = st.selectbox("Target", df.columns.tolist(), key="automl_target")
        st.session_state.automl_input_df = df

with tabs[1]:
    if st.session_state.get("automl_input_df") is None:
        st.info("Upload data in tab 1 first.")
    else:
        task = st.session_state.automl_task
        available_algos = list(model_candidates(task).keys())
        selected_algo = st.selectbox("Algorithm", available_algos, key="automl_algorithm")
        tuning_level = st.select_slider("Tuning level", options=["basic", "light", "medium", "heavy", "extreme"], value="medium", key="automl_tuning")
        scorer_options = ["f1_weighted", "accuracy"] if task == "classification" else ["neg_root_mean_squared_error", "r2"]
        scorer_name = st.selectbox("Optimization metric", scorer_options, key="automl_metric")

        if st.button("Run Auto ML"):
            try:
                df = st.session_state.automl_input_df
                target = st.session_state.automl_target
                ctx = split_context(df, target, task, test_size=0.2, random_state=42)

                pipe = build_feature_pipeline(ctx.train_df[ctx.feature_cols], "standard", "most_frequent", "mean")
                x_train = pipe.fit_transform(ctx.train_df[ctx.feature_cols], ctx.y_train)
                x_test = pipe.transform(ctx.test_df[ctx.feature_cols])
                model = clone(model_candidates(task)[selected_algo])

                cv_scores = cross_val_score(model, x_train, ctx.y_train, cv=5, scoring=scorer_name)
                model.fit(x_train, ctx.y_train)

                scorer = get_scorer(scorer_name)
                out_score = float(scorer(model, x_test, ctx.y_test))
                boot_vals = []
                n_obs = len(ctx.y_test)
                for _ in range(30):
                    idx = np.random.randint(0, n_obs, n_obs)
                    boot_vals.append(float(scorer(model, x_test[idx], ctx.y_test.iloc[idx])))

                in_metrics = evaluate_metrics(model, task, x_train, ctx.y_train)
                out_metrics = evaluate_metrics(model, task, x_test, ctx.y_test)

                summary = pd.DataFrame([
                    {
                        "model": selected_algo,
                        "metric_in_sample_mean_cv": float(cv_scores.mean()),
                        "metric_in_sample_std_cv": float(cv_scores.std()),
                        "metric_in_sample_mean_bootstrap": float(np.mean(boot_vals)),
                        "metric_in_sample_std_bootstrap": float(np.std(boot_vals)),
                        "metric_out_of_sample": out_score,
                    }
                ])

                metric_table = pd.DataFrame(
                    {
                        "metric": sorted(set(list(in_metrics.keys()) + list(out_metrics.keys()))),
                    }
                )
                metric_table["in_sample"] = metric_table["metric"].map(in_metrics)
                metric_table["out_of_sample"] = metric_table["metric"].map(out_metrics)

                st.session_state.automl_summary = summary
                st.session_state.automl_metric_table = metric_table
                st.session_state.automl_model = model
                st.session_state.automl_pipeline = pipe
                st.session_state.automl_ctx = ctx
                st.session_state.automl_x_train = x_train
                st.session_state.automl_x_test = x_test
                st.session_state.automl_feature_names = get_feature_names(pipe)
                st.success("Auto ML completed successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.info("Proposed fix: verify target column, remove invalid values, and retry with clean data.")

        st.markdown("### Training configuration")
        st.table(
            pd.DataFrame(
                {
                    "Setting": ["Mode", "Task", "Target", "Algorithm", "Tuning", "Optimization metric", "Test size", "Random state", "Preprocessing"],
                    "Value": [
                        "Auto ML",
                        st.session_state.get("automl_task"),
                        st.session_state.get("automl_target"),
                        st.session_state.get("automl_algorithm"),
                        st.session_state.get("automl_tuning"),
                        st.session_state.get("automl_metric"),
                        0.2,
                        42,
                        "impute(mean/most_frequent) + standard scale",
                    ],
                }
            )
        )

with tabs[2]:
    if st.session_state.automl_summary is None:
        st.info("Run Auto ML first in tab 2.")
    else:
        summary = st.session_state.automl_summary
        st.dataframe(summary)
        st.dataframe(st.session_state.automl_metric_table)

        st.plotly_chart(px.bar(summary, x="model", y="metric_in_sample_mean_cv", title="CV Mean"), use_container_width=True)
        st.plotly_chart(px.bar(summary, x="model", y="metric_in_sample_mean_bootstrap", title="Bootstrap Mean"), use_container_width=True)
        st.plotly_chart(px.bar(summary, x="model", y="metric_out_of_sample", title="Out-of-sample"), use_container_width=True)

        if st.session_state.automl_ctx.task_type == "classification":
            import matplotlib.pyplot as plt

            y_test = st.session_state.automl_ctx.y_test
            y_pred = st.session_state.automl_model.predict(st.session_state.automl_x_test)
            denominator = st.selectbox("Confusion matrix denominator", ["none", "true", "pred", "all"], index=0, key="automl_cm_denom")
            denom_arg = None if denominator == "none" else denominator
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=denom_arg, ax=ax)
            st.pyplot(fig)

with tabs[3]:
    if st.session_state.automl_summary is None:
        st.info("Run Auto ML first in tab 2.")
    else:
        x_test = st.session_state.automl_x_test
        x_train = st.session_state.automl_x_train
        if sparse.issparse(x_test):
            x_test = x_test.toarray()
        if sparse.issparse(x_train):
            x_train = x_train.toarray()
        y_test = st.session_state.automl_ctx.y_test
        feature_names = st.session_state.automl_feature_names

        importance = permutation_importance(st.session_state.automl_model, x_test, y_test, n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importance.importances_mean}).sort_values("importance", ascending=False)

        left, right = st.columns(2)
        with left:
            st.dataframe(imp_df.head(25))
        with right:
            st.plotly_chart(px.bar(imp_df.head(25), x="feature", y="importance", title="Permutation Importance"), use_container_width=True)

        pdp_features = st.multiselect("Columns for PDP", feature_names, default=feature_names[:1])
        if pdp_features:
            import matplotlib.pyplot as plt

            feature_idx = [feature_names.index(f) for f in pdp_features]
            fig, ax = plt.subplots(figsize=(7, 4))
            PartialDependenceDisplay.from_estimator(st.session_state.automl_model, x_test, feature_idx, ax=ax)
            st.pyplot(fig)

        if HAS_SHAP:
            st.markdown("#### SHAP")
            explainer = shap.Explainer(st.session_state.automl_model, x_train)
            shap_values = explainer(x_test[: min(100, len(x_test))])
            shap_importance = np.abs(shap_values.values).mean(axis=0)
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": shap_importance}).sort_values("mean_abs_shap", ascending=False)
            st.dataframe(shap_df.head(25))
            st.plotly_chart(px.bar(shap_df.head(25), x="feature", y="mean_abs_shap", title="SHAP Importance"), use_container_width=True)

        if HAS_LIME:
            st.markdown("#### LIME")
            class_names = [str(c) for c in np.unique(st.session_state.automl_ctx.y_train)] if st.session_state.automl_ctx.task_type == "classification" else None
            explainer = LimeTabularExplainer(x_train, feature_names=feature_names, class_names=class_names, mode="classification" if st.session_state.automl_ctx.task_type == "classification" else "regression")
            explain_idx = st.slider("LIME row index", 0, max(0, len(x_test) - 1), 0)
            if st.session_state.automl_ctx.task_type == "classification" and hasattr(st.session_state.automl_model, "predict_proba"):
                exp = explainer.explain_instance(x_test[explain_idx], st.session_state.automl_model.predict_proba)
            else:
                exp = explainer.explain_instance(x_test[explain_idx], st.session_state.automl_model.predict)
            lime_df = pd.DataFrame(exp.as_list(), columns=["feature", "weight"])
            st.dataframe(lime_df)

current_tab = st.session_state.get("automl_tab_selector", tab_labels[0])
current_idx = tab_labels.index(current_tab) if current_tab in tab_labels else 0
c_prev, c_next = st.columns(2)
with c_prev:
    if st.button("⬅️ Previous tab", disabled=current_idx == 0, key="automl_prev_tab"):
        st.session_state.automl_tab_selected = tab_labels[current_idx - 1]
        st.rerun()
with c_next:
    if st.button("Next tab ➡️", disabled=current_idx >= len(tab_labels) - 1, key="automl_next_tab"):
        st.session_state.automl_tab_selected = tab_labels[current_idx + 1]
        st.rerun()
