import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from helpers.data_utils import read_uploaded_data, split_context
from helpers.ml_utils import build_feature_pipeline, evaluate_metrics
from helpers.navigation import render_top_navigation
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()
render_top_navigation()

st.header("AI λab — Auto ML")
uploaded = st.file_uploader("Upload csv/json/xlsx", type=["csv", "json", "xlsx"], key="automl_upload")
if uploaded is not None:
    ext = uploaded.name.split(".")[-1]
    df = read_uploaded_data(uploaded, ext)
    target = st.selectbox("Target", df.columns.tolist(), key="automl_target")
    task = st.selectbox("Task", ["classification", "regression"], key="automl_task")


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

    if st.button("Run ABSOLUTE MINIMUM"):
        try:
            ctx = split_context(df, target, task, test_size=0.2, random_state=42)
            pipe = build_feature_pipeline(ctx.train_df[ctx.feature_cols], "standard", "most_frequent", "mean")
            x_train = pipe.fit_transform(ctx.train_df[ctx.feature_cols], ctx.y_train)
            x_test = pipe.transform(ctx.test_df[ctx.feature_cols])
            model = RandomForestClassifier(n_estimators=300) if task == "classification" else RandomForestRegressor(n_estimators=300)
            model.fit(x_train, ctx.y_train)
            st.json(evaluate_metrics(model, task, x_test, ctx.y_test))
            st.success("Auto ML completed successfully.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.info("Proposed fix: verify target column, remove invalid values, and retry with clean data.")
