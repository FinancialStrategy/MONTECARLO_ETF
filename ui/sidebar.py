# ui/sidebar.py

import streamlit as st
from datetime import datetime, timedelta
from config import INVESTMENT_UNIVERSE, DEFAULT_NUM_SIMULATIONS, DEFAULT_FORECAST_DAYS, DEFAULT_RISK_FREE_RATE

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Institutional Controls")

        default_etfs = ["SPY", "QQQ", "AGG", "GLD", "VNQ"]
        selected_etfs = st.multiselect(
            "Select ETFs",
            options=list(INVESTMENT_UNIVERSE.keys()),
            default=default_etfs
        )

        end_date = datetime.now()
        start_date = st.date_input("Start Date", end_date - timedelta(days=5*365), max_value=end_date)
        end_date_input = st.date_input("End Date", end_date, max_value=end_date)

        use_log_returns = st.toggle("Use Log Returns", value=False)
        covariance_method = st.selectbox("Covariance Method", ["Sample", "Ledoit-Wolf"])
        risk_free_rate = st.number_input("Risk-Free Rate", min_value=0.0, max_value=0.20, value=DEFAULT_RISK_FREE_RATE, step=0.005)

        allocation_method = st.radio(
            "Allocation Method",
            ["Equal Weight", "Optimized (Max Sharpe)", "Optimized (Min Volatility)", "Tracking Error Optimization", "Custom Weights"]
        )

        custom_weights = None
        if allocation_method == "Custom Weights" and selected_etfs:
            weights = []
            total = 0.0
            for t in selected_etfs:
                w = st.number_input(f"{t} Weight (%)", min_value=0.0, max_value=100.0, value=round(100/len(selected_etfs), 2), step=1.0)
                weights.append(w / 100.0)
                total += w
            custom_weights = weights
            if abs(total - 100) > 0.1:
                st.error(f"Weight total = {total:.2f}%")

        num_simulations = st.slider("Simulations", 1000, 50000, DEFAULT_NUM_SIMULATIONS, 1000)
        forecast_days = st.slider("Forecast Days", 21, 756, DEFAULT_FORECAST_DAYS, 21)
        initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=100000, step=5000)

        st.subheader("Black-Litterman Views")
        use_bl = st.toggle("Enable Black-Litterman", value=False)

        run_button = st.button("🚀 Run Analytics", type="primary", use_container_width=True)

    return {
        "selected_etfs": selected_etfs,
        "start_date": start_date,
        "end_date": end_date_input,
        "use_log_returns": use_log_returns,
        "covariance_method": covariance_method,
        "risk_free_rate": risk_free_rate,
        "allocation_method": allocation_method,
        "custom_weights": custom_weights,
        "num_simulations": num_simulations,
        "forecast_days": forecast_days,
        "initial_investment": initial_investment,
        "use_bl": use_bl,
        "run_button": run_button,
    }
