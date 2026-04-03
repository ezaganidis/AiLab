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
from helpers.ml_utils import build_feature_pipeline, get_feature_names, model_candidates
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

PRETTY_METRICS = {
    "accuracy": "Accuracy",
    "f1_weighted": "F1 (Weighted)",
    "precision_weighted": "Precision (Weighted)",
    "recall_weighted": "Recall (Weighted)",
    "roc_auc_ovr": "ROC AUC (OVR)",
    "r2": "R²",
    "neg_root_mean_squared_error": "Mean Root Square Error (MRSE)",
    "neg_mean_absolute_error": "Mean Absolute Error (MAE)",
}


def pretty_metric_name(metric_key: str) -> str:
    return PRETTY_METRICS.get(metric_key, metric_key.replace("_", " ").title())


def enforce_column_types(df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col, kind in type_map.items():
        if col not in out.columns:
            continue
        if kind == "numerical":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif kind == "date":
            out[col] = pd.to_datetime(out[col], errors="coerce")
        else:
            out[col] = out[col].astype("string")
    return out


for key in [
    "automl_summary",
    "automl_model",
    "automl_pipeline",
    "automl_ctx",
    "automl_x_train",
    "automl_x_test",
    "automl_metric_table",
    "automl_feature_names",
    "automl_trained_models",
]:
    if key not in st.session_state:
        st.session_state[key] = None

st.subheader("Auto ML")
tab_labels = ["1) Import & Setup", "2) Modeling", "3) Results", "4) XAI"]
if "automl_tab_selected" not in st.session_state:
    st.session_state.automl_tab_selected = tab_labels[0]
tabs = st.tabs(tab_labels, default=st.session_state.automl_tab_selected)

with tabs[0]:
    uploaded = st.file_uploader("Upload csv/json/xlsx", type=["csv", "json", "xlsx"], key="automl_upload")
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1]
        df = read_uploaded_data(uploaded, ext)
        st.session_state.automl_input_df = df

        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Task", ["classification", "regression"], key="automl_task")
        with c2:
            st.selectbox("Target", df.columns.tolist(), key="automl_target")

        st.markdown("### Column type setup")
        type_options = ["categorical", "numerical", "date"]
        inferred = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                inferred[col] = "numerical"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                inferred[col] = "date"
            else:
                inferred[col] = "categorical"
        if "automl_col_types" not in st.session_state:
            st.session_state.automl_col_types = inferred.copy()
        rows = []
        for col in df.columns:
            t = st.selectbox(
                f"Type for {col}",
                type_options,
                index=type_options.index(st.session_state.automl_col_types.get(col, inferred[col])),
                key=f"automl_col_type_{col}",
            )
            st.session_state.automl_col_types[col] = t
            rows.append({"column": col, "type": t})
        st.dataframe(pd.DataFrame(rows), width="content")

with tabs[1]:
    if st.session_state.get("automl_input_df") is None:
        st.info("Upload data in tab 1 first.")
    else:
        task = st.session_state.automl_task
        all_algos = list(model_candidates(task).keys())
        defaults = ["logistic_regression", "random_forest"] if task == "classification" else ["linear_regression", "random_forest"]
        default_algos = [a for a in defaults if a in all_algos] or all_algos[:2]
        st.multiselect("Algorithms", all_algos, default=default_algos, key="automl_algorithms")

        st.select_slider("Tuning level", options=["basic", "light", "medium", "heavy", "extreme"], value="medium", key="automl_tuning")

        metric_options = ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted", "roc_auc_ovr"] if task == "classification" else ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"]
        metric_labels = {pretty_metric_name(m): m for m in metric_options}
        selected_metric_label = st.selectbox("Optimization metric", list(metric_labels.keys()), key="automl_metric_label")
        st.session_state.automl_metric = metric_labels[selected_metric_label]

        selected_report = st.multiselect(
            "Metrics to report in results",
            metric_options,
            default=[st.session_state.automl_metric],
            format_func=pretty_metric_name,
            key="automl_report_metrics",
        )
        if st.session_state.automl_metric not in selected_report:
            selected_report = [st.session_state.automl_metric] + selected_report

        if st.button("Run Auto ML"):
            try:
                df = enforce_column_types(st.session_state.automl_input_df, st.session_state.get("automl_col_types", {}))
                target = st.session_state.automl_target
                ctx = split_context(df, target, task, test_size=0.2, random_state=42)

                pipe = build_feature_pipeline(ctx.train_df[ctx.feature_cols], "standard", "most_frequent", "mean")
                x_train = pipe.fit_transform(ctx.train_df[ctx.feature_cols], ctx.y_train)
                x_test = pipe.transform(ctx.test_df[ctx.feature_cols])

                trained_models = {}
                summary_rows = []
                best_model = None
                best_cv = -np.inf

                for algo in st.session_state.automl_algorithms:
                    base = clone(model_candidates(task)[algo])
                    base.fit(x_train, ctx.y_train)
                    trained_models[algo] = base

                    for metric_key in selected_report:
                        scorer = get_scorer(metric_key)
                        cv_scores = cross_val_score(clone(model_candidates(task)[algo]), x_train, ctx.y_train, cv=5, scoring=metric_key)
                        out_score = float(scorer(base, x_test, ctx.y_test))
                        n_obs = len(ctx.y_test)
                        boot_vals = []
                        for _ in range(30):
                            idx = np.random.randint(0, n_obs, n_obs)
                            boot_vals.append(float(scorer(base, x_test[idx], ctx.y_test.iloc[idx])))

                        summary_rows.append(
                            {
                                "model": algo,
                                "metric": pretty_metric_name(metric_key),
                                "metric_key": metric_key,
                                "metric_in_sample_mean_cv": float(np.mean(cv_scores)),
                                "metric_in_sample_std_cv": float(np.std(cv_scores)),
                                "metric_in_sample_mean_bootstrap": float(np.mean(boot_vals)),
                                "metric_in_sample_std_bootstrap": float(np.std(boot_vals)),
                                "metric_out_of_sample": out_score,
                            }
                        )

                        if metric_key == st.session_state.automl_metric and float(np.mean(cv_scores)) > best_cv:
                            best_cv = float(np.mean(cv_scores))
                            best_model = base

                summary = pd.DataFrame(summary_rows)
                st.session_state.automl_summary = summary
                st.session_state.automl_model = best_model
                st.session_state.automl_trained_models = trained_models
                st.session_state.automl_pipeline = pipe
                st.session_state.automl_ctx = ctx
                st.session_state.automl_x_train = x_train
                st.session_state.automl_x_test = x_test
                st.session_state.automl_feature_names = get_feature_names(pipe)
                st.success("Auto ML completed successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

        st.markdown("### Training configuration")
        mid = st.columns([1, 2, 1])[1]
        with mid:
            st.table(
                pd.DataFrame(
                    {
                        "Setting": ["Mode", "Task", "Target", "Algorithms", "Tuning", "Optimization metric", "Reported metrics", "Test size", "Random state"],
                        "Value": [
                            "Auto ML",
                            st.session_state.get("automl_task"),
                            st.session_state.get("automl_target"),
                            ", ".join(st.session_state.get("automl_algorithms", [])),
                            st.session_state.get("automl_tuning"),
                            pretty_metric_name(st.session_state.get("automl_metric", "")),
                            ", ".join(pretty_metric_name(m) for m in st.session_state.get("automl_report_metrics", [])),
                            0.2,
                            42,
                        ],
                    }
                )
            )

with tabs[2]:
    if st.session_state.automl_summary is None or st.session_state.automl_summary.empty:
        st.info("Run Auto ML first in tab 2.")
    else:
        summary = st.session_state.automl_summary.copy()
        default_metric = pretty_metric_name(st.session_state.get("automl_metric", ""))
        metrics_in_results = summary["metric"].dropna().unique().tolist()
        selected_metric_view = st.multiselect("Metrics shown in table/graphs", metrics_in_results, default=[default_metric] if default_metric in metrics_in_results else metrics_in_results[:1])
        shown = summary[summary["metric"].isin(selected_metric_view)] if selected_metric_view else summary

        mid = st.columns([1, 2, 1])[1]
        with mid:
            st.dataframe(
                shown[
                    [
                        "model",
                        "metric",
                        "metric_in_sample_mean_cv",
                        "metric_in_sample_std_cv",
                        "metric_in_sample_mean_bootstrap",
                        "metric_in_sample_std_bootstrap",
                        "metric_out_of_sample",
                    ]
                ],
                width="content",
            )

        if not shown.empty:
            st.plotly_chart(px.bar(shown, x="model", y="metric_in_sample_mean_cv", color="metric", barmode="group", title="CV Mean by Model/Metric"), width="stretch")
            st.plotly_chart(px.bar(shown, x="model", y="metric_in_sample_mean_bootstrap", color="metric", barmode="group", title="Bootstrap Mean by Model/Metric"), width="stretch")
            st.plotly_chart(px.bar(shown, x="model", y="metric_out_of_sample", color="metric", barmode="group", title="Out-of-Sample by Model/Metric"), width="stretch")
            st.plotly_chart(px.scatter(shown, x="metric_in_sample_mean_cv", y="metric_out_of_sample", color="metric", text="model", title="CV Mean vs Out-of-Sample"), width="stretch")

        if st.session_state.automl_ctx.task_type == "classification" and st.session_state.automl_model is not None:
            import matplotlib.pyplot as plt

            y_test = st.session_state.automl_ctx.y_test
            y_pred = st.session_state.automl_model.predict(st.session_state.automl_x_test)
            denominator = st.selectbox("Confusion matrix denominator", ["none", "true", "pred", "all"], index=0, key="automl_cm_denom")
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=None if denominator == "none" else denominator, ax=ax)
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
            st.dataframe(imp_df.head(25), width="content")
        with right:
            st.plotly_chart(px.bar(imp_df.head(25), x="feature", y="importance", title="Permutation Importance"), width="stretch")

        pdp_features = st.multiselect("Columns for PDP", feature_names, default=feature_names[:1])
        if pdp_features:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4))
            PartialDependenceDisplay.from_estimator(st.session_state.automl_model, x_test, [feature_names.index(f) for f in pdp_features], ax=ax)
            st.pyplot(fig)

        if HAS_SHAP:
            st.markdown("#### SHAP")
            explainer = shap.Explainer(st.session_state.automl_model, x_train)
            shap_values = explainer(x_test[: min(100, len(x_test))])
            shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
            st.dataframe(shap_df.head(25), width="content")
            st.plotly_chart(px.bar(shap_df.head(25), x="feature", y="mean_abs_shap", title="SHAP Importance"), width="stretch")

        if HAS_LIME:
            st.markdown("#### LIME")
            class_names = [str(c) for c in np.unique(st.session_state.automl_ctx.y_train)] if st.session_state.automl_ctx.task_type == "classification" else None
            explainer = LimeTabularExplainer(x_train, feature_names=feature_names, class_names=class_names, mode="classification" if st.session_state.automl_ctx.task_type == "classification" else "regression")
            explain_idx = st.slider("LIME row index", 0, max(0, len(x_test) - 1), 0)
            pred_fn = st.session_state.automl_model.predict_proba if st.session_state.automl_ctx.task_type == "classification" and hasattr(st.session_state.automl_model, "predict_proba") else st.session_state.automl_model.predict
            exp = explainer.explain_instance(x_test[explain_idx], pred_fn)
            st.dataframe(pd.DataFrame(exp.as_list(), columns=["feature", "weight"]), width="content")

# Bottom navigation (next all the way right)
current_idx = tab_labels.index(st.session_state.get("automl_tab_selected", tab_labels[0]))
nav_cols = st.columns([1, 8, 1])
with nav_cols[0]:
    if st.button("⬅️ Previous tab", disabled=current_idx == 0, key="automl_prev_tab"):
        st.session_state.automl_tab_selected = tab_labels[current_idx - 1]
        st.rerun()
with nav_cols[2]:
    if st.button("Next tab ➡️", disabled=current_idx >= len(tab_labels) - 1, key="automl_next_tab"):
        st.session_state.automl_tab_selected = tab_labels[current_idx + 1]
        st.rerun()
