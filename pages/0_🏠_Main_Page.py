import streamlit as st

from helpers.config import LOGO_PATH
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()

st.header("Welcome to AI λab")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=220)

st.markdown(
    """
AI λab is a production-style machine learning workbench.

### What this app does
- **Auto ML** for quick baseline runs.
- **Custom ML** for full control over preprocessing, selection, tuning, and evaluation.
- **Predict** to score new datasets using saved artifacts.

### Who it is for
- Data analysts and scientists.
- Teams sharing local folders while preparing cloud deployment.
- ML practitioners who want reproducible workflows.

### How to use
1. Open **Custom ML** and configure your training flow.
2. Save both model bundle and pipeline artifacts.
3. Use **Predict** with saved assets for new data.
"""
)
