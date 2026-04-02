import streamlit as st

from helpers.config import LOGO_PATH, ensure_app_dirs
from helpers.logging_utils import setup_logger
from helpers.navigation import render_top_navigation
from helpers.state import initialize_session_state
from helpers.style import set_app_style

st.set_page_config(page_title="ML λab", layout="wide", initial_sidebar_state="collapsed")
ensure_app_dirs()
logger = setup_logger(verbose=True)
initialize_session_state()
set_app_style()
render_top_navigation()

logger.info("App loaded")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=180)
st.header("Main Page")
