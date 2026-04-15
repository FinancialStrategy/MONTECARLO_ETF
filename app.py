from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from config import APP_SUBTITLE, APP_TITLE, INVESTMENT_UNIVERSE
from core.black_litterman import BlackLittermanModel
from core.data_loader import DataLoader
from core.monte_carlo import MonteCarloEngine
from core.optimization import PortfolioOptimizer
from core.regime import RegimeDetector
from core.relative_risk import beta_alpha, information_ratio, relative_var_cvar_es, tracking_error
from core.reporting import allocation_table, benchmark_probability_table, percentile_table
from core.risk import risk_summary_table, rolling_relative_tail_metrics
from exports.excel_export import build_excel_report
from exports.pdf_export import build_pdf_report, dataframe_to_png
from theme import apply_theme
from ui.charts import (
    category_pie_chart,
    cumulative_relative_chart,
    efficient_frontier_chart,
    monte_carlo_paths_chart,
    regime_chart,
    rolling_relative_tail_chart,
    terminal_distribution_chart,
    weight_bar_chart,
)
from ui.sidebar import render_sidebar


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
apply_theme()

st.markdown(
    f"""
    <div class="hero-box">
        <div class="main-title">{APP_TITLE}</div>
        <div class="sub-title">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("".join([f'<span class="badge">{ticker}</span>' for ticker in INVESTMENT_UNIVERSE.keys()]), unsafe_allow_html=True)

state = render_sidebar()
if not state["run_button"]:
    st.info("Select your controls from the sidebar and run the platform.")
    st.stop()

selected_etfs = state["selected_etfs"]
if not selected_etfs:
    st.error("Select at least one ETF.")
    st.stop()

loader = DataLoader(selected_etfs, state["start_date"], state["end_date"], use_log_returns=state["use_log_returns"])
with st.spinner("Downloading market data and building the analytics engine..."):
    prices = loader.fetch_prices()
    data = loader.compute_returns(prices)

returns = data["returns"]
benchmark_returns_df = data["benchmark_returns"]
current_prices = data["current_prices"]

optimizer = PortfolioOptimizer(returns=returns, risk_free_rate=state["risk_free_rate"], covariance_method=state["covariance_method"])
benchmark_choice = st.selectbox("Relative Analytics Benchmark", ["SPY", "QQQ"], index=0)
benchmark_series = benchmark_returns_df[benchmark_choice].copy()
benchmark_series.name = benchmark_choice

posterior_returns = None
posterior_cov = None
bl_output = None
if state["use_bl"]:
    st.subheader("Black-Litterman Views")
    market_weights = np.repeat(1 / len(selected_etfs), len(selected_etfs))
    bl = BlackLittermanModel(cov_matrix=optimizer.cov_matrix, market_weights=market_weights, risk_aversion=2.5, tau=0.05)
    P, Q, confidences = [], [], []
    for i in range(state["num_views"]):
        st.markdown(f"**View {i + 1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            left_asset = st.selectbox(f"Asset A {i + 1}", selected_etfs, key=f"left_{i}")
        with col2:
            right_asset = st.selectbox(f"Asset B {i + 1}", ["None"] + selected_etfs, key=f"right_{i}")
        with col3:
            view_return = st.slider(f"Annual View {i + 1} (%)", -20.0, 20.0, 3.0, 0.5, key=f"view_{i}")
        conf = st.slider(f"Confidence {i + 1}", 0.05, 0.95, 0.60, 0.05, key=f"conf_{i}")

        row = np.zeros(len(selected_etfs))
        row[selected_etfs.index(left_asset)] = 1.0
        if right_asset != "None":
            row[selected_etfs.index(right_asset)] = -1.0
        P.append(row)
        Q.append(view_return / 100.0)
        confidences.append(conf)

    bl_output = bl.posterior(np.array(P), np.array(Q), np.array(confidences))
    posterior_returns = bl_output["posterior_returns"]
    posterior_cov = bl_output["posterior_cov"]
    st.dataframe(
        pd.DataFrame({
            "Ticker": posterior_returns.index,
            "Equilibrium Return": bl_output["equilibrium_returns"].values,
            "Posterior Return": posterior_returns.values,
        }),
        use_container_width=True,
        hide_index=True,
    )

if state["allocation_method"] == "Equal Weight":
    weights = np.repeat(1 / len(selected_etfs), len(selected_etfs))
elif state["allocation_method"] == "Optimized (Max Sharpe)":
    weights = optimizer.optimize("max_sharpe", mean_returns=posterior_returns, cov_matrix=posterior_cov)
elif state["allocation_method"] == "Optimized (Min Volatility)":
    weights = optimizer.optimize("min_volatility", mean_returns=posterior_returns, cov_matrix=posterior_cov)
elif state["allocation_method"] == "Tracking Error Optimization":
    weights = optimizer.optimize_tracking_error(benchmark_series)
elif state["allocation_method"] == "Active Risk Budgeting":
    weights = optimizer.optimize_active_risk_budget(benchmark_series, target_tracking_error=state["target_te"])
else:
    weights = np.array(state["custom_weights"], dtype=float)

alloc_df = allocation_table(selected_etfs, weights, INVESTMENT_UNIVERSE)
portfolio_returns = returns @ weights
portfolio_returns.name = "Portfolio"

mean_for_mc = posterior_returns if posterior_returns is not None else returns.mean()
cov_for_mc = posterior_cov if posterior_cov is not None else optimizer.cov_matrix
mc = MonteCarloEngine(mean_returns=mean_for_mc, cov_matrix=cov_for_mc, num_simulations=state["num_simulations"], forecast_days=state["forecast_days"])
with st.spinner("Running Monte Carlo simulation and building reports..."):
    sim = mc.run(weights, state["initial_investment"])

risk_df = risk_summary_table(portfolio_returns, risk_free_rate=state["risk_free_rate"])
relative_tail = relative_var_cvar_es(portfolio_returns, benchmark_series)
rolling_relative = rolling_relative_tail_metrics(portfolio_returns, benchmark_series, window=63, alpha=0.95)
te = tracking_error(portfolio_returns, benchmark_series)
ir = information_ratio(portfolio_returns, benchmark_series)
beta, alpha = beta_alpha(portfolio_returns, benchmark_series)
regime = RegimeDetector(portfolio_returns, window=63).detect()
bench_prob_df = benchmark_probability_table(sim["final_values"], state["initial_investment"])
pct_df = percentile_table(sim["final_values"], state["initial_investment"])

frontier_hist = optimizer.benchmark_relative_frontier(benchmark_series, points=120)
frontier_bl = None
if posterior_returns is not None and posterior_cov is not None:
    temp_optimizer = PortfolioOptimizer(returns=returns, risk_free_rate=state["risk_free_rate"], covariance_method=state["covariance_method"])
    frontier_bl_rows = []
    for _ in range(120):
        w = np.random.random(len(selected_etfs))
        w /= w.sum()
        ret, vol, shr = temp_optimizer.portfolio_stats(w, posterior_returns, posterior_cov)
        frontier_bl_rows.append({"return": ret * 100, "volatility": vol * 100, "sharpe": shr, "weights": w})
    frontier_bl = pd.DataFrame(frontier_bl_rows)

summary_tab, mc_tab, opt_tab, rel_tab, regime_tab, report_tab = st.tabs([
    "Executive Summary",
    "Monte Carlo",
    "Optimization",
    "Relative Risk",
    "Regime Detection",
    "Downloads",
])

with summary_tab:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Expected Terminal Value", f"${sim['expected_value']:,.2f}")
    with m2:
        st.metric("Median Terminal Value", f"${sim['median_value']:,.2f}")
    with m3:
        st.metric("Tracking Error", f"{te*100:.2f}%")
    with m4:
        st.metric("Information Ratio", f"{ir:.3f}" if not np.isnan(ir) else "N/A")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(weight_bar_chart(alloc_df), use_container_width=True)
    with c2:
        st.plotly_chart(category_pie_chart(alloc_df), use_container_width=True)

    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

with mc_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(monte_carlo_paths_chart(sim["portfolio_values"], state["initial_investment"]), use_container_width=True)
    with c2:
        st.plotly_chart(terminal_distribution_chart(sim["final_values"], state["initial_investment"]), use_container_width=True)
    st.dataframe(bench_prob_df, use_container_width=True, hide_index=True)
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

with opt_tab:
    st.subheader("Benchmark-Relative Efficient Frontier")
    st.plotly_chart(efficient_frontier_chart(frontier_hist, "tracking_error", "active_return", "information_ratio", "Benchmark-Relative Efficient Frontier"), use_container_width=True)
    if frontier_bl is not None:
        st.subheader("Black-Litterman Efficient Frontier")
        st.plotly_chart(efficient_frontier_chart(frontier_bl, "volatility", "return", "sharpe", "Posterior Efficient Frontier"), use_container_width=True)

with rel_tab:
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Beta", f"{beta:.3f}" if not np.isnan(beta) else "N/A")
    with a2:
        st.metric("Alpha", f"{alpha*100:.2f}%" if not np.isnan(alpha) else "N/A")
    with a3:
        st.metric("Relative VaR", f"{relative_tail['relative_var']*100:.2f}%")
    with a4:
        st.metric("Relative CVaR", f"{relative_tail['relative_cvar']*100:.2f}%")

    st.plotly_chart(cumulative_relative_chart(portfolio_returns, benchmark_series), use_container_width=True)
    if not rolling_relative.empty:
        st.plotly_chart(rolling_relative_tail_chart(rolling_relative), use_container_width=True)

with regime_tab:
    st.plotly_chart(regime_chart(regime), use_container_width=True)
    st.dataframe(regime.tail(50), use_container_width=True)

with report_tab:
    excel_tables = {
        "Allocation": alloc_df,
        "Risk": risk_df,
        "RelativeRisk": pd.DataFrame({
            "Metric": ["Tracking Error", "Information Ratio", "Beta", "Alpha", "Relative VaR", "Relative CVaR", "Relative ES"],
            "Value": [te, ir, beta, alpha, relative_tail["relative_var"], relative_tail["relative_cvar"], relative_tail["relative_es"]],
        }),
        "BenchmarkProbabilities": bench_prob_df,
        "Percentiles": pct_df,
        "RollingRelative": rolling_relative.reset_index() if not rolling_relative.empty else pd.DataFrame(),
    }
    excel_file = build_excel_report(excel_tables)

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        image_paths.append(dataframe_to_png(alloc_df.head(10), "Top Allocation Weights", os.path.join(tmpdir, "allocation.png")))
        image_paths.append(dataframe_to_png(risk_df, "Risk Summary", os.path.join(tmpdir, "risk.png")))
        rel_export_df = pd.DataFrame({
            "Metric": ["Tracking Error", "Information Ratio", "Beta", "Alpha", "Relative VaR", "Relative CVaR", "Relative ES"],
            "Value": [te, ir, beta, alpha, relative_tail["relative_var"], relative_tail["relative_cvar"], relative_tail["relative_es"]],
        })
        image_paths.append(dataframe_to_png(rel_export_df, "Relative Risk Summary", os.path.join(tmpdir, "relative.png")))

        summary_lines = [
            f"Selected ETFs: {', '.join(selected_etfs)}",
            f"Initial Investment: ${state['initial_investment']:,.2f}",
            f"Expected Terminal Value: ${sim['expected_value']:,.2f}",
            f"Tracking Error vs {benchmark_choice}: {te*100:.2f}%",
            f"Information Ratio: {ir:.3f}" if not np.isnan(ir) else "Information Ratio: N/A",
            f"Relative CVaR: {relative_tail['relative_cvar']:.4f}",
            f"Allocation Method: {state['allocation_method']}",
            f"Black-Litterman Enabled: {'Yes' if state['use_bl'] else 'No'}",
        ]
        pdf_file = build_pdf_report(summary_lines, image_paths=image_paths)

        st.download_button(
            "Download Excel Report",
            data=excel_file,
            file_name="institutional_portfolio_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_file,
            file_name="institutional_portfolio_report.pdf",
            mime="application/pdf",
        )

    st.dataframe(risk_df, use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame([relative_tail]), use_container_width=True, hide_index=True)
