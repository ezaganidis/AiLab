import streamlit as st

from helpers.config import ensure_app_dirs
from helpers.logging_utils import setup_logger
from helpers.state import initialize_session_state
from helpers.style import set_app_style

st.set_page_config(page_title="AI λab", layout="wide", initial_sidebar_state="collapsed")
ensure_app_dirs()
logger = setup_logger(verbose=True)
initialize_session_state()
set_app_style()

logger.info("App loaded")

nav = st.navigation(
    [
        st.Page("pages/0_🏠_Main_Page.py", title="Main Page", icon="🏠"),
        st.Page("pages/1_🤖_Auto_ML.py", title="Auto ML", icon="🤖"),
        st.Page("pages/2_🧪_Custom_ML.py", title="Custom ML", icon="🧪"),
        st.Page("pages/3_🔮_Predict.py", title="Predict", icon="🔮"),
    ],
    position="top",
)
nav.run()
