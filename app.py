# app.py

import streamlit as st
import numpy as np
import pandas as pd

from config import INVESTMENT_UNIVERSE
from theme import apply_theme

from core.data_loader import DataLoader
from core.optimization import PortfolioOptimizer
from core.monte_carlo import MonteCarloEngine
from core.black_litterman import BlackLittermanModel
from core.regime import RegimeDetector
from core.relative_risk import tracking_error, information_ratio, beta_alpha, relative_var_cvar_es
from core.reporting import allocation_table, benchmark_probability_table, percentile_table
from core.risk import summary_risk_table

from ui.sidebar import render_sidebar
from ui.charts import weight_bar_chart, category_pie_chart, monte_carlo_paths_chart, terminal_distribution_chart, regime_chart

from exports.excel_export import build_excel_report
from exports.pdf_export import build_pdf_report

st.set_page_config(
    page_title="Institutional Portfolio Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

st.markdown(
    """
    <div class="hero-box">
        <div class="main-title">📊 Institutional Portfolio Analytics Platform</div>
        <div class="sub-title">
            Modular portfolio construction, Black-Litterman, tracking error optimization,
            regime detection, relative tail risk, and downloadable institutional reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

state = render_sidebar()

if not state["run_button"]:
    st.info("Select your controls from the sidebar and run the platform.")
    st.stop()

selected_etfs = state["selected_etfs"]
if not selected_etfs:
    st.error("Select at least one ETF.")
    st.stop()

loader = DataLoader(
    selected_etfs,
    state["start_date"],
    state["end_date"],
    use_log_returns=state["use_log_returns"]
)

prices = loader.fetch_prices()
data = loader.compute_returns(prices)

returns = data["returns"]
benchmark_returns_df = data["benchmark_returns"]
current_prices = data["current_prices"]

optimizer = PortfolioOptimizer(
    returns=returns,
    risk_free_rate=state["risk_free_rate"],
    covariance_method=state["covariance_method"]
)

if state["allocation_method"] == "Equal Weight":
    weights = np.repeat(1 / len(selected_etfs), len(selected_etfs))
elif state["allocation_method"] == "Optimized (Max Sharpe)":
    weights = optimizer.optimize("max_sharpe")
elif state["allocation_method"] == "Optimized (Min Volatility)":
    weights = optimizer.optimize("min_volatility")
elif state["allocation_method"] == "Tracking Error Optimization":
    bmk = benchmark_returns_df["SPY"].copy()
    bmk.name = "SPY"
    weights = optimizer.optimize_tracking_error(bmk)
else:
    weights = np.array(state["custom_weights"], dtype=float)

# Optional Black-Litterman overlay
if state["use_bl"]:
    st.subheader("Black-Litterman Views")
    st.caption("Example setup: one or more tactical views with confidence sliders.")

    num_views = st.slider("Number of Views", 1, min(3, len(selected_etfs)), 1)
    market_weights = np.repeat(1 / len(selected_etfs), len(selected_etfs))

    bl = BlackLittermanModel(
        cov_matrix=optimizer.cov_matrix,
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.05
    )

    P = []
    Q = []
    confidences = []

    for i in range(num_views):
        st.markdown(f"**View {i+1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            left_asset = st.selectbox(f"Asset A {i+1}", selected_etfs, key=f"left_{i}")
        with col2:
            right_asset = st.selectbox(f"Asset B {i+1}", ["None"] + selected_etfs, key=f"right_{i}")
        with col3:
            view_return = st.slider(f"Expected Relative/Absolute Annual View {i+1} (%)", -20.0, 20.0, 3.0, 0.5, key=f"view_{i}")

        conf = st.slider(f"Confidence {i+1}", 0.05, 0.95, 0.60, 0.05, key=f"conf_{i}")

        row = np.zeros(len(selected_etfs))
        row[selected_etfs.index(left_asset)] = 1.0
        if right_asset != "None":
            row[selected_etfs.index(right_asset)] = -1.0

        P.append(row)
        Q.append(view_return / 100.0)
        confidences.append(conf)

    posterior = bl.posterior(np.array(P), np.array(Q), np.array(confidences))
    st.write("Posterior Expected Returns")
    st.dataframe(
        posterior["posterior_returns"].rename("Posterior Return").reset_index().rename(columns={"index": "Ticker"}),
        use_container_width=True,
        hide_index=True
    )

alloc_df = allocation_table(selected_etfs, weights, INVESTMENT_UNIVERSE)

portfolio_returns = returns @ weights
portfolio_returns.name = "Portfolio"

mc = MonteCarloEngine(
    mean_returns=returns.mean(),
    cov_matrix=optimizer.cov_matrix,
    num_simulations=state["num_simulations"],
    forecast_days=state["forecast_days"]
)

sim = mc.run(weights, state["initial_investment"])

benchmark_choice = st.selectbox("Relative Analytics Benchmark", ["SPY", "QQQ"], index=0)
benchmark_series = benchmark_returns_df[benchmark_choice].copy()
benchmark_series.name = benchmark_choice

risk_df = summary_risk_table(portfolio_returns, risk_free_rate=state["risk_free_rate"])
relative_tail = relative_var_cvar_es(portfolio_returns, benchmark_series)
te = tracking_error(portfolio_returns, benchmark_series)
ir = information_ratio(portfolio_returns, benchmark_series)
beta, alpha = beta_alpha(portfolio_returns, benchmark_series)

regime = RegimeDetector(portfolio_returns, window=63).detect()

bench_prob_df = benchmark_probability_table(sim["final_values"], state["initial_investment"])
pct_df = percentile_table(sim["final_values"], state["initial_investment"])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "Monte Carlo",
    "Optimization",
    "Relative Risk",
    "Regime Detection",
    "Downloads"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(weight_bar_chart(alloc_df), use_container_width=True)
    with c2:
        st.plotly_chart(category_pie_chart(alloc_df), use_container_width=True)

    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(monte_carlo_paths_chart(sim["portfolio_values"], state["initial_investment"]), use_container_width=True)
    with col2:
        st.plotly_chart(terminal_distribution_chart(sim["final_values"], state["initial_investment"]), use_container_width=True)

    st.dataframe(bench_prob_df, use_container_width=True, hide_index=True)
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Risk Summary")
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

with tab4:
    st.metric("Tracking Error", f"{te*100:.2f}%")
    st.metric("Information Ratio", f"{ir:.3f}" if not np.isnan(ir) else "N/A")
    st.metric("Beta", f"{beta:.3f}" if not np.isnan(beta) else "N/A")
    st.metric("Alpha", f"{alpha*100:.2f}%" if not np.isnan(alpha) else "N/A")

    rel_df = pd.DataFrame({
        "Metric": ["Relative VaR", "Relative CVaR", "Relative ES"],
        "Value": [
            relative_tail["relative_var"],
            relative_tail["relative_cvar"],
            relative_tail["relative_es"]
        ]
    })
    st.dataframe(rel_df, use_container_width=True, hide_index=True)

with tab5:
    st.plotly_chart(regime_chart(regime), use_container_width=True)
    st.dataframe(regime.tail(50), use_container_width=True)

with tab6:
    excel_tables = {
        "Allocation": alloc_df,
        "Risk": risk_df,
        "RelativeRisk": pd.DataFrame({
            "Metric": ["Tracking Error", "Information Ratio", "Beta", "Alpha", "Relative VaR", "Relative CVaR", "Relative ES"],
            "Value": [te, ir, beta, alpha, relative_tail["relative_var"], relative_tail["relative_cvar"], relative_tail["relative_es"]]
        }),
        "BenchmarkProbabilities": bench_prob_df,
        "Percentiles": pct_df
    }

    excel_file = build_excel_report(excel_tables)

    summary_lines = [
        "Institutional Portfolio Analytics Report",
        f"Selected ETFs: {', '.join(selected_etfs)}",
        f"Initial Investment: ${state['initial_investment']:,.2f}",
        f"Expected Terminal Value: ${sim['expected_value']:,.2f}",
        f"Tracking Error vs {benchmark_choice}: {te*100:.2f}%",
        f"Information Ratio: {ir:.3f}" if not np.isnan(ir) else "Information Ratio: N/A",
        f"Relative CVaR: {relative_tail['relative_cvar']:.4f}",
    ]
    pdf_file = build_pdf_report(summary_lines)

    st.download_button(
        "Download Excel Report",
        data=excel_file,
        file_name="institutional_portfolio_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.download_button(
        "Download PDF Report",
        data=pdf_file,
        file_name="institutional_portfolio_report.pdf",
        mime="application/pdf"
    )
