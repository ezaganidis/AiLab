import streamlit as st

from helpers.data_utils import read_uploaded_data
from helpers.ml_utils import list_saved_models, list_saved_pipelines, load_model_bundle, load_pipeline
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()

model_files = list_saved_models()
if not model_files:
    st.info("No saved model bundles found")
    st.stop()

selected_model = st.selectbox("Select model bundle", model_files, key="predict_selected_model")
use_pipeline = st.checkbox("Use separately saved pipeline", key="predict_use_pipeline")
selected_pipeline = None
if use_pipeline:
    pipeline_files = list_saved_pipelines()
    if not pipeline_files:
        st.warning("No saved pipelines available")
    else:
        selected_pipeline = st.selectbox("Select saved pipeline", pipeline_files, key="predict_selected_pipeline")

upload = st.file_uploader("Upload data for inference", type=["csv", "json", "xlsx"])
if upload is not None:
    try:
        ext = upload.name.split(".")[-1]
        infer_df = read_uploaded_data(upload, ext)
        bundle = load_model_bundle(selected_model)
        pipeline = load_pipeline(selected_pipeline) if selected_pipeline else bundle["pipeline"]
        x = pipeline.transform(infer_df)

        selector = bundle.get("selector")
        if selector is not None:
            x = selector.transform(x)

        out = infer_df.copy()
        out["prediction"] = bundle["model"].predict(x)
        st.dataframe(out.head(200))
        st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        st.success("Prediction completed successfully.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info("Proposed fix: verify column names/types match training schema and retry.")
