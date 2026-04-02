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

        # Auto ML widget state
        "automl_target": None,
        "automl_task": "classification",
        # Custom ML widget state
        "custom_source": "Upload file",
        "custom_file_type": "csv",
        "custom_target": None,
        "custom_task_type": "classification",
        "custom_test_size": 0.2,
        "custom_random_state": 42,
        "custom_num_fill": "mean",
        "custom_cat_fill": "most_frequent",
        "custom_scaler": "standard",
        "custom_selection_method": "manual",
        "custom_selection_k": 5,
        "custom_metric": None,
        "custom_folds": 5,
        "custom_tuning_level": "medium",
        "custom_imbalance": "none",
        "custom_use_stacking": False,
        "custom_stacking_models": [],
        # Predict widget state
        "predict_selected_model": None,
        "predict_use_pipeline": False,
        "predict_selected_pipeline": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
