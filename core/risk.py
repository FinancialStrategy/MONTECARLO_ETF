# core/risk.py

"""
Risk analytics module for the Institutional Portfolio Analytics Platform.

Includes:
- annualized return
- annualized volatility
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- maximum drawdown
- historical VaR / CVaR
- parametric VaR
- summary risk table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def _to_series(returns) -> pd.Series:
    """
    Safely convert input returns into a clean pandas Series.
    """
    if isinstance(returns, pd.Series):
        s = returns.copy()
    elif isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise ValueError("DataFrame input must contain exactly one column for risk calculations.")
        s = returns.iloc[:, 0].copy()
    else:
        s = pd.Series(returns)

    s = pd.to_numeric(s, errors="coerce").dropna()
    return s


def safe_divide(a: float, b: float) -> float:
    """
    Safe division helper.
    Returns np.nan when denominator is zero or invalid.
    """
    try:
        if b is None or np.isnan(b) or b == 0:
            return np.nan
    except Exception:
        return np.nan
    return a / b


def annualize_return(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized geometric return from periodic returns.
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan

    growth = (1.0 + r).prod()
    n = len(r)

    if n == 0 or growth <= 0:
        return np.nan

    return growth ** (periods_per_year / n) - 1.0


def annualize_volatility(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized volatility from periodic returns.
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan

    return r.std(ddof=1) * np.sqrt(periods_per_year)


def downside_volatility(returns, mar_daily: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute downside volatility relative to a minimum acceptable return (MAR).
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan

    downside = np.minimum(r - mar_daily, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)


def cumulative_wealth_index(returns) -> pd.Series:
    """
    Convert return series into cumulative wealth index.
    """
    r = _to_series(returns)
    if r.empty:
        return pd.Series(dtype=float)

    return (1.0 + r).cumprod()


def drawdown_series(returns) -> pd.Series:
    """
    Compute drawdown series from return stream.
    """
    wealth = cumulative_wealth_index(returns)
    if wealth.empty:
        return pd.Series(dtype=float)

    running_max = wealth.cummax()
    dd = wealth / running_max - 1.0
    return dd


def max_drawdown(returns) -> float:
    """
    Compute maximum drawdown from periodic returns.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return np.nan
    return dd.min()


def sharpe_ratio(returns, risk_free_rate: float = 0.03, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized Sharpe ratio.
    """
    ann_ret = annualize_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year=periods_per_year)
    return safe_divide(ann_ret - risk_free_rate, ann_vol)


def sortino_ratio(returns, risk_free_rate: float = 0.03, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute annualized Sortino ratio.
    """
    ann_ret = annualize_return(returns, periods_per_year=periods_per_year)
    dvol = downside_volatility(
        returns,
        mar_daily=risk_free_rate / periods_per_year,
        periods_per_year=periods_per_year,
    )
    return safe_divide(ann_ret - risk_free_rate, dvol)


def calmar_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """
    Compute Calmar ratio.
    """
    ann_ret = annualize_return(returns, periods_per_year=periods_per_year)
    mdd = max_drawdown(returns)
    return safe_divide(ann_ret, abs(mdd))


def historical_var_cvar(returns, alpha: float = 0.95) -> tuple[float, float]:
    """
    Compute historical VaR and CVaR at the specified confidence level.

    Returns:
        (VaR, CVaR)
    Both values are in return space, not currency space.
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan, np.nan

    q = r.quantile(1.0 - alpha)
    tail = r[r <= q]
    cvar = tail.mean() if len(tail) > 0 else q

    return q, cvar


def parametric_var(returns, alpha: float = 0.95) -> float:
    """
    Compute Gaussian / parametric VaR.
    """
    r = _to_series(returns)
    if r.empty:
        return np.nan

    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(1.0 - alpha)

    return mu + z * sigma


def basic_distribution_stats(returns) -> dict:
    """
    Compute additional distribution diagnostics.
    """
    r = _to_series(returns)
    if r.empty:
        return {
            "mean_periodic": np.nan,
            "median_periodic": np.nan,
            "std_periodic": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "min_return": np.nan,
            "max_return": np.nan,
        }

    return {
        "mean_periodic": r.mean(),
        "median_periodic": r.median(),
        "std_periodic": r.std(ddof=1),
        "skewness": r.skew(),
        "kurtosis": r.kurt(),
        "min_return": r.min(),
        "max_return": r.max(),
    }


def summary_risk_table(returns, risk_free_rate: float = 0.03, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Create a summary risk table for reporting and dashboard use.
    """
    r = _to_series(returns)

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
                    "Historical VaR 95%",
                    "Historical CVaR 95%",
                    "Parametric VaR 95%",
                    "Mean Periodic Return",
                    "Median Periodic Return",
                    "Periodic Std Dev",
                    "Skewness",
                    "Kurtosis",
                    "Minimum Periodic Return",
                    "Maximum Periodic Return",
                ],
                "Value": [np.nan] * 16,
            }
        )

    hist_var95, hist_cvar95 = historical_var_cvar(r, alpha=0.95)
    param_var95 = parametric_var(r, alpha=0.95)
    dist_stats = basic_distribution_stats(r)

    table = pd.DataFrame(
        {
            "Metric": [
                "Annual Return",
                "Annual Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Calmar Ratio",
                "Maximum Drawdown",
                "Historical VaR 95%",
                "Historical CVaR 95%",
                "Parametric VaR 95%",
                "Mean Periodic Return",
                "Median Periodic Return",
                "Periodic Std Dev",
                "Skewness",
                "Kurtosis",
                "Minimum Periodic Return",
                "Maximum Periodic Return",
            ],
            "Value": [
                annualize_return(r, periods_per_year=periods_per_year),
                annualize_volatility(r, periods_per_year=periods_per_year),
                sharpe_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
                sortino_ratio(r, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year),
                calmar_ratio(r, periods_per_year=periods_per_year),
                max_drawdown(r),
                hist_var95,
                hist_cvar95,
                param_var95,
                dist_stats["mean_periodic"],
                dist_stats["median_periodic"],
                dist_stats["std_periodic"],
                dist_stats["skewness"],
                dist_stats["kurtosis"],
                dist_stats["min_return"],
                dist_stats["max_return"],
            ],
        }
    )

    return table


def format_risk_table_for_display(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional helper for display formatting.
    """
    df = risk_df.copy()

    pct_metrics = {
        "Annual Return",
        "Annual Volatility",
        "Maximum Drawdown",
        "Historical VaR 95%",
        "Historical CVaR 95%",
        "Parametric VaR 95%",
        "Mean Periodic Return",
        "Median Periodic Return",
        "Periodic Std Dev",
        "Minimum Periodic Return",
        "Maximum Periodic Return",
    }

    formatted_values = []
    for _, row in df.iterrows():
        metric = row["Metric"]
        value = row["Value"]

        if pd.isna(value):
            formatted_values.append("N/A")
        elif metric in pct_metrics:
            formatted_values.append(f"{value * 100:.4f}%")
        else:
            formatted_values.append(f"{value:.4f}")

    df["Formatted Value"] = formatted_values
    return df
