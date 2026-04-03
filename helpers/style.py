from pathlib import Path

import streamlit as st


def add_fixed_footer(text: str = "Developed by Emmanouil Zaganidis") -> None:
    st.markdown(
        f"""
        <style>
          .block-container {{
            padding-bottom: 3.2rem;
          }}
          #app-footer {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            z-index: 9999;
            text-align: center;
            padding: 0.6rem 0;
            font-size: 0.85rem;
            opacity: 0.85;
            backdrop-filter: blur(6px);
            background: rgba(255, 255, 255, 0.6);
            border-top: 1px solid rgba(0, 0, 0, 0.08);
          }}
          @media (prefers-color-scheme: dark) {{
            #app-footer {{
              background: rgba(0, 0, 0, 0.35);
              border-top: 1px solid rgba(255, 255, 255, 0.10);
            }}
          }}
        </style>
        <div id="app-footer">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def set_app_style() -> None:
    logo_path = Path("images") / "logo_app.svg"
    icon_path = Path("images") / "logo_app.svg"

    if hasattr(st, "logo") and logo_path.exists():
        st.logo(
            str(logo_path),
            size="large",
            icon_image=str(icon_path) if icon_path.exists() else None,
            link=None,
        )

    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 4.25rem;
            padding-bottom: 1rem;
            max-width: 98vw;
          }
          div[data-testid="stVerticalBlock"] { gap: 0.75rem; }
          h1, h2, h3 { margin-bottom: 0.25rem; }
          .small { font-size: 0.9rem; opacity: 0.85; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    add_fixed_footer("Developed by Emmanouil Zaganidis")
