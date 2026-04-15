# theme.py

import streamlit as st

def apply_theme():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #EAF2FF;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.0rem;
            color: #AFC2DC;
            margin-bottom: 1rem;
        }
        .hero-box {
            background: linear-gradient(135deg, #0F172A 0%, #13233C 60%, #173E69 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .soft-card {
            background: linear-gradient(135deg, rgba(18,25,41,0.96), rgba(24,33,54,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 0.9rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
