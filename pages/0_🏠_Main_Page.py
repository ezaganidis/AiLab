import streamlit as st

from helpers.config import LOGO_PATH
from helpers.state import initialize_session_state
from helpers.style import set_app_style

initialize_session_state()
set_app_style()

st.header("Welcome to AI λab")
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=220)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Modes")
    st.write("Auto ML, Custom ML, Predict")

with c2:
    st.subheader("Workflow")
    st.write("Import → EDA → Feature Engineering → Modeling → XAI")

with c3:
    c3.header("What is this app for?")
    c3.write(
        """
        This application is a practical Machine Learning laboratory for end-to-end experimentation:
        - importing datasets,
        - performing EDA,
        - applying preprocessing and feature engineering,
        - training and tuning multiple Machine Learning models, and
        - producing explainability and evaluation outputs.
        """
    )
    c3.header("Who is it for?")
    c3.write(
        """
        - Students and researchers who want a structured Machine Learning workflow
        - Practitioners who want fast experimentation with reproducible outputs
        - Anyone learning how to compare models, metrics, and explainability methods
        """
    )

with c4:
    c4.header("Professors")
    c4.write(
        """
        **Periklis Gogas**
        **Theofilos Papadimitriou**
        """
    )

    c4.header("Developer")
    c4.write("**Emmanouil Zaganidis**")
