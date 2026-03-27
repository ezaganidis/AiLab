import streamlit as st


def initialize_session_state() -> None:
    defaults = {
        "raw_df": None,
        "ctx": None,
        "feature_pipeline": None,
        "selected_features": None,
        "models_summary": None,
        "best_model": None,
        "best_model_name": None,
        "x_train_ready": None,
        "x_test_ready": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
