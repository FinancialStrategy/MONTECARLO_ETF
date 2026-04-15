# core/risk.py

import numpy as np
import pandas as pd
from scipy.stats import norm
from config import TRADING_DAYS
from core.utils import safe_div, drawdown_series

def annualize_return(returns):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    growth = (1 + returns).prod()
    n = len(returns)
    if growth <= 0 or n == 0:
        return np.nan
    return growth ** (TRADING_DAYS / n) - 1

def annualize_volatility(returns):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

def downside_volatility(returns, mar_daily=0.0):
    returns = pd.Series(returns).dropna()
    downside = np.minimum(returns - mar_daily, 0)
    return np.sqrt(np.mean(downside**2)) * np.sqrt(TRADING_DAYS)

def sharpe_ratio(returns, risk_free_rate=0.03):
    ann_ret = annualize_return(returns)
    ann_vol = annualize_volatility(returns)
    return safe_div(ann_ret - risk_free_rate, ann_vol)

def sortino_ratio(returns, risk_free_rate=0.03):
    ann_ret = annualize_return(returns)
    dvol = downside_volatility(returns, mar_daily=risk_free_rate / TRADING_DAYS)
    return safe_div(ann_ret - risk_free_rate, dvol)

def calmar_ratio(returns):
    ann_ret = annualize_return(returns)
    dd = max_drawdown(returns)
    return safe_div(ann_ret, abs(dd))

def max_drawdown(returns):
    wealth = (1 + pd.Series(returns).dropna()).cumprod()
    dd = drawdown_series(wealth)
    return dd.min()

def historical_var_cvar(returns, alpha=0.95):
    r = pd.Series(returns).dropna()
    q = r.quantile(1 - alpha)
    tail = r[r <= q]
    cvar = tail.mean() if len(tail) else q
    return q, cvar

def parametric_var(returns, alpha=0.95):
    r = pd.Series(returns).dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(1 - alpha)
    return mu + z * sigma

def summary_risk_table(returns, risk_free_rate=0.03):
    var95, cvar95 = historical_var_cvar(returns, alpha=0.95)
    return pd.DataFrame({
        "Metric": [
            "Annual Return", "Annual Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Calmar Ratio", "Maximum Drawdown",
            "Historical VaR 95%", "Historical CVaR 95%", "Parametric VaR 95%"
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
            parametric_var(returns, alpha=0.95)
        ]
    })
