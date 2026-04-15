# theme.py
"""
Institutional UI Theme for Streamlit Application
Provides reusable styling for consistent branding.
"""

import streamlit as st
from pathlib import Path
import base64


def _load_image_as_base64(image_path: str) -> str:
    """
    Convert an image to a base64 string for embedding in CSS/HTML.
    """
    path = Path(image_path)
    if not path.exists():
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def apply_theme(logo_path: str = None):
    """
    Apply institutional styling and optional logo to the Streamlit app.

    Parameters
    ----------
    logo_path : str, optional
        Path to a logo image located in the assets directory.
    """

    logo_base64 = _load_image_as_base64(logo_path) if logo_path else ""

    st.markdown(
        f"""
        <style>
        :root {{
            --primary-color: #0B1F3A;
            --secondary-color: #1F4E79;
            --accent-color: #4EA8FF;
            --background-color: #F5F7FA;
            --card-background: #FFFFFF;
            --text-color: #1A1A1A;
            --border-radius: 14px;
        }}

        .main-title {{
            font-size: 2.6rem;
            font-weight: 800;
            color: var(--primary-color);
            margin-bottom: 0.2rem;
        }}

        .sub-title {{
            font-size: 1.1rem;
            color: var(--secondary-color);
            margin-bottom: 1.5rem;
        }}

        .hero-box {{
            background: linear-gradient(135deg, #0B1F3A, #1F4E79);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            color: white;
            margin-bottom: 1.5rem;
        }}

        .kpi-card {{
            background: var(--card-background);
            border-radius: var(--border-radius);
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 1rem;
        }}

        .kpi-label {{
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .kpi-value {{
            font-size: 1.6rem;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .section-box {{
            background: var(--card-background);
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #e0e6ed;
        }}

        .logo-container {{
            text-align: center;
            margin-bottom: 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display logo in sidebar if provided
    if logo_base64:
        st.sidebar.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" width="180">
            </div>
            """,
            unsafe_allow_html=True,
        )
