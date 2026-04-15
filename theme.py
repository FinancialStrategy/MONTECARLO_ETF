import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text-main: #EAF2FF;
            --text-soft: #AFC2DC;
            --border: rgba(255,255,255,0.08);
        }
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.0rem;
            color: var(--text-soft);
            margin-bottom: 1rem;
        }
        .hero-box {
            background: linear-gradient(135deg, #0F172A 0%, #13233C 60%, #173E69 100%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .soft-card {
            background: linear-gradient(135deg, rgba(18,25,41,0.96), rgba(24,33,54,0.96));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .badge {
            display: inline-block;
            background: rgba(78,168,255,0.14);
            color: #D4E9FF;
            border: 1px solid rgba(78,168,255,0.25);
            border-radius: 999px;
            padding: 0.24rem 0.6rem;
            font-size: 0.76rem;
            font-weight: 600;
            margin: 0.15rem 0.15rem 0.15rem 0;
        }
        div[data-testid="stMetric"] {
            background: rgba(17,24,39,0.35);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.8rem;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
