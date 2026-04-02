import streamlit as st


def render_top_navigation() -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.page_link("app.py", label="🏠 Main Page")
    with c2:
        st.page_link("pages/1_🤖_Auto_ML.py", label="🤖 Auto ML")
    with c3:
        st.page_link("pages/2_🧪_Custom_ML.py", label="🧪 Custom ML")
    with c4:
        st.page_link("pages/3_🔮_Predict.py", label="🔮 Predict")
    st.markdown("---")
