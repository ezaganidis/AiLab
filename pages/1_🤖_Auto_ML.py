import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

from helpers.data_utils import read_uploaded_data, split_context
from helpers.ml_utils import build_feature_pipeline, evaluate_metrics, get_feature_names
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()

if "automl_summary" not in st.session_state:
    st.session_state.automl_summary = None
if "automl_model" not in st.session_state:
    st.session_state.automl_model = None
if "automl_pipeline" not in st.session_state:
    st.session_state.automl_pipeline = None
if "automl_ctx" not in st.session_state:
    st.session_state.automl_ctx = None
if "automl_x_train" not in st.session_state:
    st.session_state.automl_x_train = None
if "automl_x_test" not in st.session_state:
    st.session_state.automl_x_test = None

tabs = st.tabs(["1) Import & Setup", "2) Run", "3) Results", "4) XAI"])

with tabs[0]:
    uploaded = st.file_uploader("Upload csv/json/xlsx", type=["csv", "json", "xlsx"], key="automl_upload")
    if uploaded is not None:
        ext = uploaded.name.split(".")[-1]
        df = read_uploaded_data(uploaded, ext)
        target = st.selectbox("Target", df.columns.tolist(), key="automl_target")
        task = st.selectbox("Task", ["classification", "regression"], key="automl_task")
        st.session_state.automl_input_df = df

        st.subheader("Training configuration")
        st.json(
            {
                "mode": "Auto ML",
                "task": task,
                "target": target,
                "algorithm": "RandomForest (default)",
                "test_size": 0.2,
                "random_state": 42,
                "numeric_imputation": "mean",
                "categorical_imputation": "most_frequent",
                "scaling": "standard",
            }
        )

with tabs[1]:
    if st.session_state.get("automl_input_df") is None:
        st.info("Upload data in tab 1 first.")
    else:
        if st.button("Run ABSOLUTE MINIMUM"):
            try:
                df = st.session_state.automl_input_df
                target = st.session_state.automl_target
                task = st.session_state.automl_task
                ctx = split_context(df, target, task, test_size=0.2, random_state=42)
                pipe = build_feature_pipeline(ctx.train_df[ctx.feature_cols], "standard", "most_frequent", "mean")
                x_train = pipe.fit_transform(ctx.train_df[ctx.feature_cols], ctx.y_train)
                x_test = pipe.transform(ctx.test_df[ctx.feature_cols])
                model = RandomForestClassifier(n_estimators=300, random_state=42) if task == "classification" else RandomForestRegressor(n_estimators=300, random_state=42)
                model.fit(x_train, ctx.y_train)

                in_metrics = evaluate_metrics(model, task, x_train, ctx.y_train)
                out_metrics = evaluate_metrics(model, task, x_test, ctx.y_test)
                out_main = out_metrics.get("f1_weighted", list(out_metrics.values())[0]) if task == "classification" else out_metrics.get("rmse", list(out_metrics.values())[0])

                summary = pd.DataFrame(
                    [
                        {
                            "model": "automl_random_forest",
                            "metric_in_sample_mean_cv": np.nan,
                            "metric_in_sample_std_cv": np.nan,
                            "metric_in_sample_mean_bootstrap": np.nan,
                            "metric_in_sample_std_bootstrap": np.nan,
                            "metric_out_of_sample": out_main,
                            "all_in_sample_metrics": json.dumps(in_metrics),
                            "all_out_sample_metrics": json.dumps(out_metrics),
                            "best_params": json.dumps({"n_estimators": 300, "random_state": 42}),
                        }
                    ]
                )

                st.session_state.automl_summary = summary
                st.session_state.automl_model = model
                st.session_state.automl_pipeline = pipe
                st.session_state.automl_ctx = ctx
                st.session_state.automl_x_train = x_train
                st.session_state.automl_x_test = x_test
                st.success("Auto ML completed successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.info("Proposed fix: verify target column, remove invalid values, and retry with clean data.")

with tabs[2]:
    if st.session_state.automl_summary is None:
        st.info("Run Auto ML first in tab 2.")
    else:
        summary = st.session_state.automl_summary
        st.dataframe(summary)
        st.plotly_chart(px.bar(summary, x="model", y="metric_out_of_sample", title="Out-of-sample metric"), use_container_width=True)

        row = summary.iloc[0]
        st.write("In-sample metrics")
        st.json(json.loads(row["all_in_sample_metrics"]))
        st.write("Out-of-sample metrics")
        st.json(json.loads(row["all_out_sample_metrics"]))

        if st.session_state.automl_ctx.task_type == "classification":
            import matplotlib.pyplot as plt

            y_test = st.session_state.automl_ctx.y_test
            y_pred = st.session_state.automl_model.predict(st.session_state.automl_x_test)
            denominator = st.selectbox("Confusion matrix denominator", ["none", "true", "pred", "all"], index=0, key="automl_cm_denom")
            denom_arg = None if denominator == "none" else denominator
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize=denom_arg, ax=ax)
            st.pyplot(fig)

        if st.session_state.automl_ctx.task_type == "regression":
            y_test = st.session_state.automl_ctx.y_test
            y_pred = st.session_state.automl_model.predict(st.session_state.automl_x_test)
            reg_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
            reg_df["residual"] = reg_df["actual"] - reg_df["predicted"]
            st.plotly_chart(px.scatter(reg_df, x="actual", y="predicted", title="Predicted vs Actual"), use_container_width=True)
            st.plotly_chart(px.histogram(reg_df, x="residual", nbins=40, title="Residual Distribution"), use_container_width=True)

with tabs[3]:
    if st.session_state.automl_summary is None:
        st.info("Run Auto ML first in tab 2.")
    else:
        x_test = st.session_state.automl_x_test
        if sparse.issparse(x_test):
            x_test = x_test.toarray()
        y_test = st.session_state.automl_ctx.y_test
        feature_names = get_feature_names(st.session_state.automl_pipeline)

        importance = permutation_importance(st.session_state.automl_model, x_test, y_test, n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importance.importances_mean}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp_df.head(25), x="feature", y="importance", title="Permutation Importance"), use_container_width=True)

        if hasattr(st.session_state.automl_model, "feature_importances_"):
            native = pd.DataFrame({"feature": feature_names, "importance": st.session_state.automl_model.feature_importances_}).sort_values("importance", ascending=False)
            st.plotly_chart(px.bar(native.head(25), x="feature", y="importance", title="Model Feature Importance"), use_container_width=True)

        if not imp_df.empty:
            import matplotlib.pyplot as plt

            top_feature = imp_df.iloc[0]["feature"]
            fig, ax = plt.subplots(figsize=(6, 4))
            PartialDependenceDisplay.from_estimator(st.session_state.automl_model, x_test, [feature_names.index(top_feature)], ax=ax)
            st.pyplot(fig)
