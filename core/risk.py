from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import TRADING_DAYS
from core.utils import drawdown_series, safe_div


def annualize_return(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    growth = (1 + returns).prod()
    n = len(returns)
    if n == 0 or growth <= 0:
        return np.nan
    return float(growth ** (TRADING_DAYS / n) - 1)


def annualize_volatility(returns: pd.Series) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def downside_volatility(returns: pd.Series, mar_daily: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    downside = np.minimum(returns - mar_daily, 0.0)
    return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    return safe_div(annualize_return(returns) - risk_free_rate, annualize_volatility(returns))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    return safe_div(annualize_return(returns) - risk_free_rate, downside_volatility(returns, risk_free_rate / TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + pd.Series(returns).dropna()).cumprod()
    return float(drawdown_series(wealth).min())


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    return safe_div(annualize_return(returns), abs(mdd))


def historical_var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = pd.Series(returns).dropna()
    q = float(r.quantile(1 - alpha))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return q, cvar


def parametric_var(returns: pd.Series, alpha: float = 0.95) -> float:
    r = pd.Series(returns).dropna()
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    z = float(norm.ppf(1 - alpha))
    return mu + z * sigma


def rolling_relative_tail_metrics(portfolio_returns: pd.Series, benchmark_returns: pd.Series, window: int = 63, alpha: float = 0.95) -> pd.DataFrame:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    active = aligned["portfolio"] - aligned["benchmark"]

    rows = []
    for i in range(window, len(active) + 1):
        win = active.iloc[i - window:i]
        q = float(win.quantile(1 - alpha))
        tail = win[win <= q]
        cvar = float(tail.mean()) if len(tail) else q
        rows.append({"date": active.index[i - 1], "relative_var": q, "relative_cvar": cvar, "relative_es": cvar})

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["relative_var", "relative_cvar", "relative_es"])


def risk_summary_table(returns: pd.Series, risk_free_rate: float = 0.03) -> pd.DataFrame:
    var95, cvar95 = historical_var_cvar(returns, 0.95)
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
            ],
            "Value": [
                annualize_return(returns),
                annualize_volatility(returns),
                sharpe_ratio(returns, risk_free_rate),
                sortino_ratio(returns, risk_free_rate),
                calmar_ratio(returns),
                max_drawdown(returns),
                var95,
                cvar95,
                parametric_var(returns, 0.95),
            ],
        }
    )
