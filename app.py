# app.py

"""
Institutional Portfolio Analytics Platform
Main Streamlit application entry point.

Features:
- modular architecture
- Black-Litterman views + confidence sliders
- tracking error optimization
- Monte Carlo simulation
- benchmark-relative analytics
- rolling relative VaR / CVaR / ES
- regime detection
- Excel / PDF report exports
- institutional UI styling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import INVESTMENT_UNIVERSE
from theme import apply_theme

from core.data_loader import DataLoader
from core.optimization import PortfolioOptimizer
from core.monte_carlo import MonteCarloEngine
from core.black_litterman import BlackLittermanModel
from core.regime import RegimeDetector
from core.relative_risk import (
    tracking_error,
    information_ratio,
    beta_alpha,
    relative_var_cvar_es,
)
from core.reporting import (
    allocation_table,
    benchmark_probability_table,
    percentile_table,
)
from core.risk import (
    risk_summary_table,
    rolling_relative_tail_metrics,
)

from ui.sidebar import render_sidebar
from ui.charts import (
    weight_bar_chart,
    category_pie_chart,
    monte_carlo_paths_chart,
    terminal_distribution_chart,
    regime_chart,
)

from exports.excel_export import build_excel_report
from exports.pdf_export import build_pdf_report


# =========================================================
# Streamlit page config
# =========================================================
st.set_page_config(
    page_title="Institutional Portfolio Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Helper functions
# =========================================================
def validate_selected_etfs(selected_etfs: list[str]) -> None:
    """
    Validate ETF selections before running the platform.
    """
    if not selected_etfs:
        st.error("Please select at least one ETF.")
        st.stop()

    invalid = [t for t in selected_etfs if t not in INVESTMENT_UNIVERSE]
    if invalid:
        st.error(f"Invalid ETF selections detected: {invalid}")
        st.stop()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights safely.
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0 or np.isnan(total):
        return np.repeat(1.0 / len(weights), len(weights))
    return weights / total


def render_hero_section() -> None:
    """
    Render top hero/header section.
    """
    st.markdown(
        """
        <div class="hero-box">
            <div class="main-title">📊 Institutional Portfolio Analytics Platform</div>
            <div class="sub-title">
                Modular portfolio construction, Black-Litterman posterior analytics,
                tracking-error optimization, regime detection, benchmark-relative tail risk,
                Monte Carlo simulation, and institutional reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_summary_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series | None,
    sim_results: dict,
    initial_investment: float,
) -> None:
    """
    Render summary KPI row.
    """
    risk_df = risk_summary_table(portfolio_returns)
    risk_map = dict(zip(risk_df["Metric"], risk_df["Value"]))

    expected_terminal = sim_results.get("expected_value", np.nan)
    prob_profit = np.mean(np.asarray(sim_results["final_values"]) > initial_investment) * 100

    if benchmark_returns is not None and not benchmark_returns.empty:
        te = tracking_error(portfolio_returns, benchmark_returns)
        ir = information_ratio(portfolio_returns, benchmark_returns)
    else:
        te = np.nan
        ir = np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Annual Return",
            f"{risk_map.get('Annual Return', np.nan) * 100:.2f}%"
            if pd.notna(risk_map.get("Annual Return", np.nan))
            else "N/A",
        )

    with c2:
        st.metric(
            "Annual Volatility",
            f"{risk_map.get('Annual Volatility', np.nan) * 100:.2f}%"
            if pd.notna(risk_map.get("Annual Volatility", np.nan))
            else "N/A",
        )

    with c3:
        st.metric(
            "Sharpe Ratio",
            f"{risk_map.get('Sharpe Ratio', np.nan):.3f}"
            if pd.notna(risk_map.get("Sharpe Ratio", np.nan))
            else "N/A",
        )

    with c4:
        st.metric(
            "Expected Terminal Value",
            f"${expected_terminal:,.2f}" if pd.notna(expected_terminal) else "N/A",
        )

    with c5:
        st.metric("Probability of Profit", f"{prob_profit:.2f}%")

    c6, c7, c8 = st.columns(3)
    with c6:
        st.metric("Tracking Error", f"{te * 100:.2f}%" if pd.notna(te) else "N/A")
    with c7:
        st.metric("Information Ratio", f"{ir:.3f}" if pd.notna(ir) else "N/A")
    with c8:
        st.metric(
            "Expected Max Drawdown",
            f"{np.mean(sim_results.get('max_drawdowns', [np.nan])) * 100:.2f}%"
            if "max_drawdowns" in sim_results
            else "N/A",
        )


def get_weights_from_method(
    allocation_method: str,
    selected_etfs: list[str],
    optimizer: PortfolioOptimizer,
    benchmark_series: pd.Series | None,
    custom_weights: list[float] | None,
) -> np.ndarray:
    """
    Determine portfolio weights from the selected allocation method.
    """
    n = len(selected_etfs)

    if allocation_method == "Equal Weight":
        return np.repeat(1.0 / n, n)

    if allocation_method == "Optimized (Max Sharpe)":
        return normalize_weights(optimizer.optimize("max_sharpe"))

    if allocation_method == "Optimized (Min Volatility)":
        return normalize_weights(optimizer.optimize("min_volatility"))

    if allocation_method == "Tracking Error Optimization":
        if benchmark_series is None or benchmark_series.empty:
            st.warning("Benchmark data unavailable. Falling back to equal weights.")
            return np.repeat(1.0 / n, n)
        return normalize_weights(optimizer.optimize_tracking_error(benchmark_series))

    if allocation_method == "Custom Weights":
        if custom_weights is None:
            st.warning("Custom weights unavailable. Falling back to equal weights.")
            return np.repeat(1.0 / n, n)
        return normalize_weights(np.array(custom_weights, dtype=float))

    st.warning("Unknown allocation method detected. Falling back to equal weights.")
    return np.repeat(1.0 / n, n)


def run_black_litterman_overlay(
    selected_etfs: list[str],
    optimizer: PortfolioOptimizer,
) -> dict | None:
    """
    Build Black-Litterman view UI and return posterior estimates if enabled.
    """
    st.subheader("Black-Litterman Views")
    st.caption(
        "Define tactical absolute or relative views and assign confidence levels. "
        "Posterior expected returns will be computed from the view set."
    )

    num_views = st.slider("Number of Black-Litterman Views", 1, min(5, len(selected_etfs)), 1)
    market_weights = np.repeat(1.0 / len(selected_etfs), len(selected_etfs))

    bl_model = BlackLittermanModel(
        cov_matrix=optimizer.cov_matrix,
        market_weights=market_weights,
        risk_aversion=2.5,
        tau=0.05,
    )

    P_rows = []
    Q_values = []
    confidence_values = []

    for i in range(num_views):
        st.markdown(f"### View {i + 1}")

        c1, c2, c3 = st.columns(3)
        with c1:
            left_asset = st.selectbox(
                f"Primary Asset {i + 1}",
                selected_etfs,
                key=f"bl_left_asset_{i}",
            )

        with c2:
            right_asset = st.selectbox(
                f"Relative Asset {i + 1}",
                ["None"] + selected_etfs,
                key=f"bl_right_asset_{i}",
            )

        with c3:
            view_return_pct = st.slider(
                f"Expected View Return {i + 1} (%)",
                min_value=-20.0,
                max_value=20.0,
                value=3.0,
                step=0.5,
                key=f"bl_view_return_{i}",
            )

        confidence = st.slider(
            f"Confidence {i + 1}",
            min_value=0.05,
            max_value=0.95,
            value=0.60,
            step=0.05,
            key=f"bl_confidence_{i}",
        )

        row = np.zeros(len(selected_etfs))
        row[selected_etfs.index(left_asset)] = 1.0

        if right_asset != "None":
            row[selected_etfs.index(right_asset)] = -1.0

        P_rows.append(row)
        Q_values.append(view_return_pct / 100.0)
        confidence_values.append(confidence)

    posterior = bl_model.posterior(
        np.array(P_rows, dtype=float),
        np.array(Q_values, dtype=float),
        np.array(confidence_values, dtype=float),
    )

    posterior_df = (
        posterior["posterior_returns"]
        .rename("Posterior Return")
        .reset_index()
        .rename(columns={"index": "Ticker"})
    )

    st.markdown("### Posterior Expected Returns")
    st.dataframe(posterior_df, use_container_width=True, hide_index=True)

    return posterior


def build_export_tables(
    allocation_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    rel_tail_df: pd.DataFrame,
    benchmark_prob_df: pd.DataFrame,
    percentile_df: pd.DataFrame,
    rolling_tail_df: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> dict:
    """
    Assemble all report tables for Excel export.
    """
    tables = {
        "Allocation": allocation_df,
        "RiskSummary": risk_df,
        "RelativeTailRisk": rel_tail_df,
        "BenchmarkProbabilities": benchmark_prob_df,
        "Percentiles": percentile_df,
        "RollingRelativeTail": rolling_tail_df.reset_index(),
        "Regimes": regime_df.reset_index(),
    }
    return tables


def build_pdf_summary_lines(
    selected_etfs: list[str],
    initial_investment: float,
    benchmark_choice: str,
    sim_results: dict,
    risk_df: pd.DataFrame,
    rel_tail_df: pd.DataFrame,
) -> list[str]:
    """
    Create summary lines for the PDF export.
    """
    risk_map = dict(zip(risk_df["Metric"], risk_df["Value"]))

    rel_map = {}
    if not rel_tail_df.empty and {"Metric", "Value"}.issubset(rel_tail_df.columns):
        rel_map = dict(zip(rel_tail_df["Metric"], rel_tail_df["Value"]))

    summary_lines = [
        "Institutional Portfolio Analytics Report",
        "",
        f"Selected ETFs: {', '.join(selected_etfs)}",
        f"Initial Investment: ${initial_investment:,.2f}",
        f"Expected Terminal Value: ${sim_results.get('expected_value', np.nan):,.2f}",
        f"Median Terminal Value: ${sim_results.get('median_value', np.nan):,.2f}",
        f"Annual Return: {risk_map.get('Annual Return', np.nan) * 100:.2f}%"
        if pd.notna(risk_map.get("Annual Return", np.nan))
        else "Annual Return: N/A",
        f"Annual Volatility: {risk_map.get('Annual Volatility', np.nan) * 100:.2f}%"
        if pd.notna(risk_map.get("Annual Volatility", np.nan))
        else "Annual Volatility: N/A",
        f"Sharpe Ratio: {risk_map.get('Sharpe Ratio', np.nan):.3f}"
        if pd.notna(risk_map.get("Sharpe Ratio", np.nan))
        else "Sharpe Ratio: N/A",
        f"Maximum Drawdown: {risk_map.get('Maximum Drawdown', np.nan) * 100:.2f}%"
        if pd.notna(risk_map.get("Maximum Drawdown", np.nan))
        else "Maximum Drawdown: N/A",
        f"Benchmark: {benchmark_choice}",
        f"Relative VaR: {rel_map.get('Relative VaR 95%', np.nan) * 100:.2f}%"
        if pd.notna(rel_map.get("Relative VaR 95%", np.nan))
        else "Relative VaR: N/A",
        f"Relative CVaR: {rel_map.get('Relative CVaR 95%', np.nan) * 100:.2f}%"
        if pd.notna(rel_map.get("Relative CVaR 95%", np.nan))
        else "Relative CVaR: N/A",
        f"Relative ES: {rel_map.get('Relative ES 95%', np.nan) * 100:.2f}%"
        if pd.notna(rel_map.get("Relative ES 95%", np.nan))
        else "Relative ES: N/A",
    ]

    return summary_lines


# =========================================================
# Main app
# =========================================================
def main():
    """
    Main Streamlit app.
    """
    apply_theme()
    render_hero_section()

    state = render_sidebar()

    if not state["run_button"]:
        st.info("Select your inputs in the sidebar and click Run Analytics.")
        return

    original_selected_etfs = state["selected_etfs"]
    validate_selected_etfs(original_selected_etfs)

    start_date = state["start_date"]
    end_date = state["end_date"]
    use_log_returns = state["use_log_returns"]
    covariance_method = state["covariance_method"]
    risk_free_rate = state["risk_free_rate"]
    allocation_method = state["allocation_method"]
    custom_weights = state["custom_weights"]
    num_simulations = state["num_simulations"]
    forecast_days = state["forecast_days"]
    initial_investment = state["initial_investment"]
    use_bl = state["use_bl"]

    if start_date >= end_date:
        st.error("Start date must be earlier than end date.")
        st.stop()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    with st.spinner("Downloading and preparing market data..."):
        loader = DataLoader(
            tickers=original_selected_etfs,
            start_date=start_date,
            end_date=end_date,
            use_log_returns=use_log_returns,
        )

        prices = loader.fetch_prices()
        data = loader.compute_returns(prices)

    returns = data["returns"]
    benchmark_returns_df = data["benchmark_returns"]
    current_prices = data["current_prices"]
    selected_etfs = data["valid_tickers"]

    if returns.empty:
        st.error("No return data available after cleaning. Please adjust date range or selections.")
        st.stop()

    if len(selected_etfs) == 0:
        st.error("No valid ETFs remain after price cleaning and alignment.")
        st.stop()

    removed_etfs = [t for t in original_selected_etfs if t not in selected_etfs]
    if removed_etfs:
        st.warning(
            "The following ETFs were removed due to insufficient data coverage or alignment: "
            + ", ".join(removed_etfs)
        )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    optimizer = PortfolioOptimizer(
        returns=returns,
        risk_free_rate=risk_free_rate,
        covariance_method=covariance_method,
    )

    # -----------------------------------------------------
    # Benchmark selection
    # -----------------------------------------------------
    benchmark_options = []
    if benchmark_returns_df is not None and not benchmark_returns_df.empty:
        benchmark_options = [c for c in ["SPY", "QQQ"] if c in benchmark_returns_df.columns]

    if len(benchmark_options) == 0:
        benchmark_choice = None
        benchmark_series = None
    else:
        benchmark_choice = st.selectbox(
            "Relative Analytics Benchmark",
            options=benchmark_options,
            index=0,
        )
        benchmark_series = benchmark_returns_df[benchmark_choice].copy()
        benchmark_series.name = benchmark_choice

    # -----------------------------------------------------
    # Portfolio weights
    # -----------------------------------------------------
    weights = get_weights_from_method(
        allocation_method=allocation_method,
        selected_etfs=selected_etfs,
        optimizer=optimizer,
        benchmark_series=benchmark_series,
        custom_weights=custom_weights,
    )

    # -----------------------------------------------------
    # Optional Black-Litterman overlay
    # -----------------------------------------------------
    posterior = None
    if use_bl:
        posterior = run_black_litterman_overlay(selected_etfs, optimizer)

    # -----------------------------------------------------
    # Allocation report
    # -----------------------------------------------------
    allocation_df = allocation_table(selected_etfs, weights, INVESTMENT_UNIVERSE)

    # -----------------------------------------------------
    # Portfolio returns
    # -----------------------------------------------------
    portfolio_returns = returns @ weights
    portfolio_returns.name = "Portfolio"

    # -----------------------------------------------------
    # Monte Carlo
    # -----------------------------------------------------
    mc_engine = MonteCarloEngine(
        mean_returns=returns.mean(),
        cov_matrix=optimizer.cov_matrix,
        num_simulations=num_simulations,
        forecast_days=forecast_days,
    )

    with st.spinner("Running Monte Carlo simulation..."):
        sim_results = mc_engine.run(weights, initial_investment)

    # -----------------------------------------------------
    # Core analytics
    # -----------------------------------------------------
    risk_df = risk_summary_table(
        portfolio_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=252,
        confidence=0.95,
    )

    if benchmark_series is not None and not benchmark_series.empty:
        relative_tail = relative_var_cvar_es(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_series,
            confidence=0.95,
        )
        rel_tail_df = pd.DataFrame(
            {
                "Metric": [
                    "Relative VaR 95%",
                    "Relative CVaR 95%",
                    "Relative ES 95%",
                ],
                "Value": [
                    relative_tail["relative_var"],
                    relative_tail["relative_cvar"],
                    relative_tail["relative_es"],
                ],
            }
        )

        rolling_tail_df = rolling_relative_tail_metrics(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_series,
            window=63,
            confidence=0.95,
        )

        te = tracking_error(portfolio_returns, benchmark_series)
        ir = information_ratio(portfolio_returns, benchmark_series)
        beta, alpha = beta_alpha(portfolio_returns, benchmark_series)
    else:
        rel_tail_df = pd.DataFrame(columns=["Metric", "Value"])
        rolling_tail_df = pd.DataFrame(
            columns=[
                "rolling_relative_var",
                "rolling_relative_cvar",
                "rolling_relative_es",
            ]
        )
        te = np.nan
        ir = np.nan
        beta = np.nan
        alpha = np.nan

    # -----------------------------------------------------
    # Regime detection
    # -----------------------------------------------------
    regime_detector = RegimeDetector(portfolio_returns, window=63)
    regime_df = regime_detector.detect()

    # -----------------------------------------------------
    # Reporting tables
    # -----------------------------------------------------
    benchmark_prob_df = benchmark_probability_table(
        sim_results["final_values"],
        initial_investment,
    )

    percentile_df = percentile_table(
        sim_results["final_values"],
        initial_investment,
    )

    # -----------------------------------------------------
    # Top metrics
    # -----------------------------------------------------
    render_top_summary_metrics(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_series,
        sim_results=sim_results,
        initial_investment=initial_investment,
    )

    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Executive Summary",
            "Monte Carlo",
            "Optimization & Allocation",
            "Relative Risk",
            "Regime Detection",
            "Downloads",
        ]
    )

    # =====================================================
    # Tab 1: Executive Summary
    # =====================================================
    with tab1:
        st.subheader("Executive Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(weight_bar_chart(allocation_df), use_container_width=True)

        with c2:
            st.plotly_chart(category_pie_chart(allocation_df), use_container_width=True)

        st.markdown("### Allocation Table")
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)

        st.markdown("### Risk Summary")
        st.dataframe(risk_df, use_container_width=True, hide_index=True)

        if benchmark_series is not None and not benchmark_series.empty:
            st.markdown(f"### Benchmark Relative Analytics vs {benchmark_choice}")
            rc1, rc2, rc3, rc4 = st.columns(4)

            with rc1:
                st.metric("Tracking Error", f"{te * 100:.2f}%" if pd.notna(te) else "N/A")
            with rc2:
                st.metric("Information Ratio", f"{ir:.3f}" if pd.notna(ir) else "N/A")
            with rc3:
                st.metric("Beta", f"{beta:.3f}" if pd.notna(beta) else "N/A")
            with rc4:
                st.metric("Alpha", f"{alpha * 100:.2f}%" if pd.notna(alpha) else "N/A")

            st.dataframe(rel_tail_df, use_container_width=True, hide_index=True)

        st.markdown("### Benchmark Outperformance Probabilities")
        st.dataframe(benchmark_prob_df, use_container_width=True, hide_index=True)

    # =====================================================
    # Tab 2: Monte Carlo
    # =====================================================
    with tab2:
        st.subheader("Monte Carlo Simulation")

        mc1, mc2 = st.columns(2)

        with mc1:
            st.plotly_chart(
                monte_carlo_paths_chart(sim_results["portfolio_values"], initial_investment),
                use_container_width=True,
            )

        with mc2:
            st.plotly_chart(
                terminal_distribution_chart(sim_results["final_values"], initial_investment),
                use_container_width=True,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Expected Terminal Value", f"${sim_results['expected_value']:,.2f}")
        with m2:
            st.metric("Median Terminal Value", f"${sim_results['median_value']:,.2f}")
        with m3:
            st.metric("Terminal Std. Dev.", f"${sim_results['std_value']:,.2f}")

        st.markdown("### Terminal Value Percentiles")
        st.dataframe(percentile_df, use_container_width=True, hide_index=True)

    # =====================================================
    # Tab 3: Optimization & Allocation
    # =====================================================
    with tab3:
        st.subheader("Optimization & Allocation")

        st.markdown("### Selected Portfolio Weights")
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)

        if posterior is not None:
            st.markdown("### Black-Litterman Posterior Returns")
            posterior_df = (
                posterior["posterior_returns"]
                .rename("Posterior Return")
                .reset_index()
                .rename(columns={"index": "Ticker"})
            )
            st.dataframe(posterior_df, use_container_width=True, hide_index=True)

        info_df = pd.DataFrame(
            {
                "Ticker": selected_etfs,
                "Name": [INVESTMENT_UNIVERSE[t]["name"] for t in selected_etfs],
                "Category": [INVESTMENT_UNIVERSE[t]["category"] for t in selected_etfs],
                "Current Price": [
                    current_prices[t] if t in current_prices.index else np.nan
                    for t in selected_etfs
                ],
            }
        )
        st.markdown("### ETF Universe Snapshot")
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    # =====================================================
    # Tab 4: Relative Risk
    # =====================================================
    with tab4:
        st.subheader("Benchmark-Relative Risk")

        if benchmark_series is None or benchmark_series.empty:
            st.warning("Benchmark data unavailable. Relative risk analytics cannot be computed.")
        else:
            st.markdown(f"### Relative Analytics vs {benchmark_choice}")

            rr1, rr2, rr3, rr4 = st.columns(4)
            with rr1:
                st.metric("Tracking Error", f"{te * 100:.2f}%" if pd.notna(te) else "N/A")
            with rr2:
                st.metric("Information Ratio", f"{ir:.3f}" if pd.notna(ir) else "N/A")
            with rr3:
                st.metric("Beta", f"{beta:.3f}" if pd.notna(beta) else "N/A")
            with rr4:
                st.metric("Alpha", f"{alpha * 100:.2f}%" if pd.notna(alpha) else "N/A")

            st.markdown("### Relative Tail Risk")
            st.dataframe(rel_tail_df, use_container_width=True, hide_index=True)

            st.markdown("### Rolling Relative Tail Metrics")
            st.dataframe(rolling_tail_df, use_container_width=True)

    # =====================================================
    # Tab 5: Regime Detection
    # =====================================================
    with tab5:
        st.subheader("Regime Detection")

        if regime_df is None or regime_df.empty:
            st.warning("Regime detection could not be computed for the current dataset.")
        else:
            st.plotly_chart(regime_chart(regime_df), use_container_width=True)
            st.dataframe(regime_df.tail(100), use_container_width=True)

    # =====================================================
    # Tab 6: Downloads
    # =====================================================
    with tab6:
        st.subheader("Download Institutional Reports")

        export_tables = build_export_tables(
            allocation_df=allocation_df,
            risk_df=risk_df,
            rel_tail_df=rel_tail_df,
            benchmark_prob_df=benchmark_prob_df,
            percentile_df=percentile_df,
            rolling_tail_df=rolling_tail_df,
            regime_df=regime_df,
        )

        excel_file = build_excel_report(export_tables)

        summary_lines = build_pdf_summary_lines(
            selected_etfs=selected_etfs,
            initial_investment=initial_investment,
            benchmark_choice=benchmark_choice if benchmark_choice is not None else "N/A",
            sim_results=sim_results,
            risk_df=risk_df,
            rel_tail_df=rel_tail_df,
        )
        pdf_file = build_pdf_report(summary_lines)

        st.download_button(
            label="Download Excel Report",
            data=excel_file,
            file_name="institutional_portfolio_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="institutional_portfolio_report.pdf",
            mime="application/pdf",
        )

        st.markdown("### Export Tables Preview")
        st.write("Sheets prepared for export:")
        for sheet_name in export_tables.keys():
            st.write(f"- {sheet_name}")

    st.markdown("---")
    st.caption(
        "Institutional note: This platform uses historical market data and model-based analytics. "
        "Outputs are analytical estimates and should not be interpreted as investment guarantees."
    )


# =========================================================
# Run app
# =========================================================
if __name__ == "__main__":
    main()
