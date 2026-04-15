# core/risk.py

"""
Risk analytics module for the Institutional Portfolio Analytics Platform.

This module provides:
- periodic return cleaning utilities
- annualized return and volatility
- Sharpe, Sortino, Calmar
- drawdown analytics
- historical VaR / CVaR / ES
- parametric VaR
- benchmark-relative tail risk
- rolling relative VaR / CVaR / ES
- portfolio risk summary tables

All functions are written to be deployment-safe for Streamlit Cloud.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


TRADING_DAYS = 252


# =========================================================
# Internal utilities
# =========================================================
def _to_series(x, name: str = "value") -> pd.Series:
    """
    Convert input to a clean numeric pandas Series.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} DataFrame must have exactly one column.")
        s = x.iloc[:, 0].copy()
    elif isinstance(x, (list, tuple, np.ndarray, Iterable)):
        s = pd.Series(x, name=name)
    else:
        s = pd.Series([x], name=name)

    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = name
    return s


def _align_two_series(
    left,
    right,
    left_name: str = "left",
    right_name: str = "right",
) -> pd.DataFrame:
    """
    Align two return series on common dates / indices.
    """
    s_left = _to_series(left, left_name)
    s_right = _to_series(right, right_name)

    aligned = pd.concat([s_left, s_right], axis=1, join="inner").dropna()
    aligned.columns = [left_name, right_name]
    return aligned


def safe_divide(a: float, b: float) -> float:
    """
    Safe divide returning np.nan on invalid denominator.
    """
    try:
        if b is None or pd.isna(b) or b == 0:
            return np.nan
    except Exception:
        return np.nan
    return a / b


# =========================================================
# Core return and risk analytics
# =========================================================
def annualized_return(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Geometric annualized return.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan

    growth = (1.0 + r).prod()
    n = len(r)

    if n == 0 or growth <= 0:
        return np.nan

    return growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Annualized standard deviation.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan
    return r.std(ddof=1) * np.sqrt(periods_per_year)


def downside_volatility(
    returns,
    mar_daily: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Downside volatility relative to minimum acceptable return.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan

    downside = np.minimum(r - mar_daily, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)


def cumulative_wealth_index(returns, initial_value: float = 1.0) -> pd.Series:
    """
    Build wealth index from periodic returns.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return pd.Series(dtype=float)

    wealth = initial_value * (1.0 + r).cumprod()
    wealth.name = "wealth"
    return wealth


def drawdown_series(returns) -> pd.Series:
    """
    Compute drawdown series from periodic returns.
    """
    wealth = cumulative_wealth_index(returns, initial_value=1.0)
    if wealth.empty:
        return pd.Series(dtype=float)

    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    dd.name = "drawdown"
    return dd


def maximum_drawdown(returns) -> float:
    """
    Compute maximum drawdown.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return np.nan
    return dd.min()


def sharpe_ratio(
    returns,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualized Sharpe ratio.
    """
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    return safe_divide(ann_ret - risk_free_rate, ann_vol)


def sortino_ratio(
    returns,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Annualized Sortino ratio.
    """
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    dvol = downside_volatility(
        returns,
        mar_daily=risk_free_rate / periods_per_year,
        periods_per_year=periods_per_year,
    )
    return safe_divide(ann_ret - risk_free_rate, dvol)


def calmar_ratio(
    returns,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Calmar ratio = annualized return / abs(max drawdown).
    """
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    mdd = maximum_drawdown(returns)
    return safe_divide(ann_ret, abs(mdd))


# =========================================================
# Tail risk analytics
# =========================================================
def historical_var_cvar(
    returns,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Historical VaR and CVaR from return series.

    Returns:
        (var, cvar)
    where var and cvar are return-space values.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan, np.nan

    q = r.quantile(1.0 - confidence)
    tail = r[r <= q]
    cvar = tail.mean() if len(tail) > 0 else q
    return q, cvar


def historical_expected_shortfall(
    returns,
    confidence: float = 0.95,
) -> float:
    """
    Alias for CVaR / Expected Shortfall.
    """
    _, cvar = historical_var_cvar(returns, confidence=confidence)
    return cvar


def parametric_var(
    returns,
    confidence: float = 0.95,
) -> float:
    """
    Gaussian / parametric VaR.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return np.nan

    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(1.0 - confidence)
    return mu + z * sigma


def tail_risk_summary(
    returns,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compact tail risk summary dictionary.
    """
    hist_var, hist_cvar = historical_var_cvar(returns, confidence=confidence)
    es = historical_expected_shortfall(returns, confidence=confidence)
    gauss_var = parametric_var(returns, confidence=confidence)

    return {
        "historical_var": hist_var,
        "historical_cvar": hist_cvar,
        "historical_es": es,
        "parametric_var": gauss_var,
    }


# =========================================================
# Relative / benchmark risk
# =========================================================
def active_return_series(
    portfolio_returns,
    benchmark_returns,
) -> pd.Series:
    """
    Active return = portfolio return - benchmark return.
    """
    aligned = _align_two_series(
        portfolio_returns,
        benchmark_returns,
        left_name="portfolio",
        right_name="benchmark",
    )
    if aligned.empty:
        return pd.Series(dtype=float, name="active_return")

    active = aligned["portfolio"] - aligned["benchmark"]
    active.name = "active_return"
    return active


def relative_var_cvar_es(
    portfolio_returns,
    benchmark_returns,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Relative tail risk of active returns.

    Returns:
        {
            "relative_var": ...,
            "relative_cvar": ...,
            "relative_es": ...
        }
    """
    active = active_return_series(portfolio_returns, benchmark_returns)
    if active.empty:
        return {
            "relative_var": np.nan,
            "relative_cvar": np.nan,
            "relative_es": np.nan,
        }

    rel_var, rel_cvar = historical_var_cvar(active, confidence=confidence)

    return {
        "relative_var": rel_var,
        "relative_cvar": rel_cvar,
        "relative_es": rel_cvar,
    }


def rolling_relative_tail_metrics(
    portfolio_returns,
    benchmark_returns,
    window: int = 63,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Rolling relative VaR / CVaR / ES of active returns.

    Returns a DataFrame indexed by date with columns:
    - rolling_relative_var
    - rolling_relative_cvar
    - rolling_relative_es
    """
    aligned = _align_two_series(
        portfolio_returns,
        benchmark_returns,
        left_name="portfolio",
        right_name="benchmark",
    )

    if aligned.empty or len(aligned) < window:
        return pd.DataFrame(
            columns=[
                "rolling_relative_var",
                "rolling_relative_cvar",
                "rolling_relative_es",
            ]
        )

    active = aligned["portfolio"] - aligned["benchmark"]

    results = []
    indices = []

    for i in range(window, len(active) + 1):
        sample = active.iloc[i - window:i]
        var_, cvar_ = historical_var_cvar(sample, confidence=confidence)

        results.append(
            {
                "rolling_relative_var": var_,
                "rolling_relative_cvar": cvar_,
                "rolling_relative_es": cvar_,
            }
        )
        indices.append(active.index[i - 1])

    out = pd.DataFrame(results, index=indices)
    return out


# =========================================================
# Distribution diagnostics
# =========================================================
def distribution_statistics(returns) -> Dict[str, float]:
    """
    Basic descriptive stats for returns.
    """
    r = _to_series(returns, "returns")
    if r.empty:
        return {
            "mean_periodic": np.nan,
            "median_periodic": np.nan,
            "std_periodic": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "min_periodic": np.nan,
            "max_periodic": np.nan,
        }

    return {
        "mean_periodic": r.mean(),
        "median_periodic": r.median(),
        "std_periodic": r.std(ddof=1),
        "skewness": r.skew(),
        "kurtosis": r.kurt(),
        "min_periodic": r.min(),
        "max_periodic": r.max(),
    }


# =========================================================
# Summary tables
# =========================================================
def risk_summary_table(
    returns,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Main risk summary table expected by app.py.

    This function is intentionally named `risk_summary_table`
    because your app is importing exactly that name.
    """
    r = _to_series(returns, "returns")

    if r.empty:
        return pd.DataFrame(
            {
                "Metric": [
                    "Annual Return",
                    "Annual Volatility",
                    "Sharpe Ratio",
                    "Sortino Ratio",
                    "Calmar Ratio",
                    "Maximum Drawdown",
                    "Historical VaR",
                    "Historical CVaR",
                    "Historical ES",
                    "Parametric VaR",
                    "Mean Periodic Return",
                    "Median Periodic Return",
                    "Periodic Std Dev",
                    "Skewness",
                    "Kurtosis",
                    "Minimum Periodic Return",
                    "Maximum Periodic Return",
                ],
                "Value": [np.nan] * 17,
            }
        )

    hist_var, hist_cvar = historical_var_cvar(r, confidence=confidence)
    hist_es = historical_expected_shortfall(r, confidence=confidence)
    gauss_var = parametric_var(r, confidence=confidence)
    dist = distribution_statistics(r)

    df = pd.DataFrame(
        {
            "Metric": [
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown",
                f"Historical VaR {int(confidence*100)}%",
                f"Historical CVaR {int(confidence*100)}%",
                f"Historical ES {int(confidence*100)}%",
                f"Parametric VaR {int(confidence*100)}%",
                "Mean Periodic Return",
                "Median Periodic Return",
                "Periodic Std Dev",
                "Skewness",
                "Kurtosis",
                "Minimum Periodic Return",
                "Maximum Periodic Return",
            ],
            "Value": [
                annualized_return(r, periods_per_year=periods_per_year),
                annualized_volatility(r, periods_per_year=periods_per_year),
                sharpe_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
                sortino_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
                calmar_ratio(r, periods_per_year=periods_per_year),
                maximum_drawdown(r),
                hist_var,
                hist_cvar,
                hist_es,
                gauss_var,
                dist["mean_periodic"],
                dist["median_periodic"],
                dist["std_periodic"],
                dist["skewness"],
                dist["kurtosis"],
                dist["min_periodic"],
                dist["max_periodic"],
            ],
        }
    )

    return df


def summary_risk_table(
    returns,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Backward-compatible alias in case other modules still call summary_risk_table.
    """
    return risk_summary_table(
        returns=returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        confidence=confidence,
    )


def relative_risk_summary_table(
    portfolio_returns,
    benchmark_returns,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Summary table for benchmark-relative tail risk.
    """
    rel = relative_var_cvar_es(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        confidence=confidence,
    )

    return pd.DataFrame(
        {
            "Metric": [
                f"Relative VaR {int(confidence*100)}%",
                f"Relative CVaR {int(confidence*100)}%",
                f"Relative ES {int(confidence*100)}%",
            ],
            "Value": [
                rel["relative_var"],
                rel["relative_cvar"],
                rel["relative_es"],
            ],
        }
    )


def format_risk_table_for_display(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional display formatter.
    Adds a formatted string column while preserving numeric values.
    """
    df = risk_df.copy()

    percent_like = {
        "Annual Return",
        "Annual Volatility",
        "Maximum Drawdown",
        "Mean Periodic Return",
        "Median Periodic Return",
        "Periodic Std Dev",
        "Minimum Periodic Return",
        "Maximum Periodic Return",
    }

    formatted = []
    for _, row in df.iterrows():
        metric = str(row["Metric"])
        value = row["Value"]

        if pd.isna(value):
            formatted.append("N/A")
        elif "VaR" in metric or "CVaR" in metric or "ES" in metric:
            formatted.append(f"{value * 100:.4f}%")
        elif metric in percent_like:
            formatted.append(f"{value * 100:.4f}%")
        else:
            formatted.append(f"{value:.4f}")

    df["Formatted Value"] = formatted
    return df
