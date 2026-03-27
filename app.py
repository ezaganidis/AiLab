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
Run this app from the project root with:

```bash
streamlit run app.py
```

Because sidebar navigation is disabled, use the buttons below to open each module/page.
"""
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link("pages/0_🏠_Main_Page.py", label="Main Page", icon="🏠")
with col2:
    st.page_link("pages/1_🤖_Auto_ML.py", label="Auto ML", icon="🤖")
with col3:
    st.page_link("pages/2_🧪_Custom_ML.py", label="Custom ML", icon="🧪")
with col4:
    st.page_link("pages/3_🔮_Predict.py", label="Predict", icon="🔮")

st.info("If a page does not open, make sure all files are inside the same project folder and rerun `streamlit run app.py` from that folder.")
