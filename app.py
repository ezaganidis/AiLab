import streamlit as st

from helpers.config import LOGO_PATH, ensure_app_dirs
from helpers.logging_utils import setup_logger
from helpers.state import initialize_session_state
from helpers.style import set_app_style

st.set_page_config(page_title="ML λab", layout="wide", initial_sidebar_state="collapsed")
ensure_app_dirs()
logger = setup_logger(verbose=True)
initialize_session_state()
set_app_style()

logger.info("App loaded")
st.title("AI λab")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=240)

st.markdown(
    """
Use the top page navigation to open:
- Main Page
- Auto ML
- Custom ML
- Predict

Your session data is kept in `st.session_state`, so switching pages/tabs does not reset your workflow.
"""
)
