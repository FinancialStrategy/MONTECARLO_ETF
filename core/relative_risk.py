# core/relative_risk.py

"""
Benchmark-relative risk analytics for the Institutional Portfolio Analytics Platform.

This module provides:
- alignment of portfolio and benchmark returns
- active return calculations
- tracking error
- information ratio
- beta and alpha
- relative VaR / CVaR / ES
- rolling tracking error
- rolling information ratio
- rolling beta
- helper summary tables

All functions are designed to be deployment-safe for Streamlit Cloud.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# =========================================================
# Internal utilities
# =========================================================
def _to_series(x, name: str = "value") -> pd.Series:
    """
    Safely convert input into a numeric pandas Series.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} DataFrame must have exactly one column.")
        s = x.iloc[:, 0].copy()
    else:
        s = pd.Series(x, name=name)

    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = name
    return s


def _align_series(
    portfolio_returns,
    benchmark_returns,
    portfolio_name: str = "portfolio",
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """
    Align portfolio and benchmark return series on common dates.
    """
    p = _to_series(portfolio_returns, portfolio_name)
    b = _to_series(benchmark_returns, benchmark_name)

    aligned = pd.concat([p, b], axis=1, join="inner").dropna()
    aligned.columns = [portfolio_name, benchmark_name]
    return aligned


def safe_divide(a: float, b: float) -> float:
    """
    Safe divide helper.
    """
    try:
        if b is None or pd.isna(b) or b == 0:
            return np.nan
    except Exception:
        return np.nan
    return a / b


# =========================================================
# Core active-return analytics
# =========================================================
def active_return_series(
    portfolio_returns,
    benchmark_returns,
) -> pd.Series:
    """
    Active return = portfolio return - benchmark return.
    """
    aligned = _align_series(portfolio_returns, benchmark_returns)

    if aligned.empty:
        return pd.Series(dtype=float, name="active_return")

    active = aligned["portfolio"] - aligned["benchmark"]
    active.name = "active_return"
    return active


def tracking_error(
    portfolio_returns,
    benchmark_returns,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualized tracking error.
    """
    active = active_return_series(portfolio_returns, benchmark_returns)
    if active.empty:
        return np.nan

    return active.std(ddof=1) * np.sqrt(periods_per_year)


def information_ratio(
    portfolio_returns,
    benchmark_returns,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualized information ratio.
    """
    active = active_return_series(portfolio_returns, benchmark_returns)
    if active.empty:
        return np.nan

    te = active.std(ddof=1) * np.sqrt(periods_per_year)
    active_ann = active.mean() * periods_per_year
    return safe_divide(active_ann, te)


def beta_alpha(
    portfolio_returns,
    benchmark_returns,
    periods_per_year: int = TRADING_DAYS,
) -> Tuple[float, float]:
    """
    Beta and annualized alpha vs benchmark.
    """
    aligned = _align_series(portfolio_returns, benchmark_returns)

    if aligned.empty:
        return np.nan, np.nan

    p = aligned["portfolio"]
    b = aligned["benchmark"]

    var_b = b.var()
    if pd.isna(var_b) or var_b == 0:
        return np.nan, np.nan

    beta = p.cov(b) / var_b
    alpha = (p.mean() - beta * b.mean()) * periods_per_year
    return beta, alpha


# =========================================================
# Relative tail risk
# =========================================================
def _historical_var_cvar(
    returns,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Historical VaR and CVaR helper on a return series.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan, np.nan

    q = r.quantile(1.0 - confidence)
    tail = r[r <= q]
    cvar = tail.mean() if len(tail) > 0 else q
    return q, cvar


def relative_var_cvar_es(
    portfolio_returns,
    benchmark_returns,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Relative tail risk of active returns.

    Parameters
    ----------
    portfolio_returns : array-like or pd.Series
    benchmark_returns : array-like or pd.Series
    confidence : float, default 0.95

    Returns
    -------
    dict
        {
            "relative_var": ...,
            "relative_cvar": ...,
            "relative_es": ...
        }

    This signature is intentionally aligned with app.py:
        relative_var_cvar_es(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_series,
            confidence=0.95,
        )
    """
    active = active_return_series(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    if active.empty:
        return {
            "relative_var": np.nan,
            "relative_cvar": np.nan,
            "relative_es": np.nan,
        }

    rel_var, rel_cvar = _historical_var_cvar(active, confidence=confidence)

    return {
        "relative_var": rel_var,
        "relative_cvar": rel_cvar,
        "relative_es": rel_cvar,
    }


def rolling_relative_var_cvar_es(
    portfolio_returns,
    benchmark_returns,
    window: int = 63,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Rolling relative VaR / CVaR / ES on active returns.
    """
    active = active_return_series(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    if active.empty or len(active) < window:
        return pd.DataFrame(
            columns=[
                "rolling_relative_var",
                "rolling_relative_cvar",
                "rolling_relative_es",
            ]
        )

    rows = []
    idx = []

    for i in range(window, len(active) + 1):
        sample = active.iloc[i - window:i]
        var_, cvar_ = _historical_var_cvar(sample, confidence=confidence)

        rows.append(
            {
                "rolling_relative_var": var_,
                "rolling_relative_cvar": cvar_,
                "rolling_relative_es": cvar_,
            }
        )
        idx.append(active.index[i - 1])

    out = pd.DataFrame(rows, index=idx)
    return out


# =========================================================
# Rolling relative performance diagnostics
# =========================================================
def rolling_tracking_error(
    portfolio_returns,
    benchmark_returns,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Rolling annualized tracking error.
    """
    active = active_return_series(portfolio_returns, benchmark_returns)

    if active.empty:
        return pd.Series(dtype=float, name="rolling_tracking_error")

    rte = active.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    rte.name = "rolling_tracking_error"
    return rte


def rolling_information_ratio(
    portfolio_returns,
    benchmark_returns,
    window: int = 63,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Rolling annualized information ratio.
    """
    active = active_return_series(portfolio_returns, benchmark_returns)

    if active.empty:
        return pd.Series(dtype=float, name="rolling_information_ratio")

    rolling_ann_active = active.rolling(window).mean() * periods_per_year
    rolling_te = active.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    rir = rolling_ann_active / rolling_te.replace(0, np.nan)
    rir.name = "rolling_information_ratio"
    return rir


def rolling_beta(
    portfolio_returns,
    benchmark_returns,
    window: int = 63,
) -> pd.Series:
    """
    Rolling beta of portfolio versus benchmark.
    """
    aligned = _align_series(portfolio_returns, benchmark_returns)

    if aligned.empty:
        return pd.Series(dtype=float, name="rolling_beta")

    p = aligned["portfolio"]
    b = aligned["benchmark"]

    cov = p.rolling(window).cov(b)
    var_b = b.rolling(window).var()

    beta = cov / var_b.replace(0, np.nan)
    beta.name = "rolling_beta"
    return beta


# =========================================================
# Summary tables
# =========================================================
def relative_risk_summary_table(
    portfolio_returns,
    benchmark_returns,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Build a compact relative-risk summary table.
    """
    te = tracking_error(portfolio_returns, benchmark_returns, periods_per_year=periods_per_year)
    ir = information_ratio(portfolio_returns, benchmark_returns, periods_per_year=periods_per_year)
    beta, alpha = beta_alpha(portfolio_returns, benchmark_returns, periods_per_year=periods_per_year)
    tail = relative_var_cvar_es(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        confidence=confidence,
    )

    df = pd.DataFrame(
        {
            "Metric": [
                "Tracking Error",
                "Information Ratio",
                "Beta",
                "Alpha",
                f"Relative VaR {int(confidence * 100)}%",
                f"Relative CVaR {int(confidence * 100)}%",
                f"Relative ES {int(confidence * 100)}%",
            ],
            "Value": [
                te,
                ir,
                beta,
                alpha,
                tail["relative_var"],
                tail["relative_cvar"],
                tail["relative_es"],
            ],
        }
    )
    return df


def format_relative_risk_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional formatter for display.
    """
    out = df.copy()

    formatted_values = []
    for _, row in out.iterrows():
        metric = str(row["Metric"])
        value = row["Value"]

        if pd.isna(value):
            formatted_values.append("N/A")
        elif "Tracking Error" in metric:
            formatted_values.append(f"{value * 100:.4f}%")
        elif "Alpha" == metric:
            formatted_values.append(f"{value * 100:.4f}%")
        elif "VaR" in metric or "CVaR" in metric or "ES" in metric:
            formatted_values.append(f"{value * 100:.4f}%")
        else:
            formatted_values.append(f"{value:.4f}")

    out["Formatted Value"] = formatted_values
    return out
