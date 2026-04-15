from __future__ import annotations

import numpy as np
import pandas as pd

from config import TRADING_DAYS
from core.utils import safe_div


def align_portfolio_and_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    return aligned


def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = align_portfolio_and_benchmark(portfolio_returns, benchmark_returns)
    active = aligned["portfolio"] - aligned["benchmark"]
    return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = align_portfolio_and_benchmark(portfolio_returns, benchmark_returns)
    active = aligned["portfolio"] - aligned["benchmark"]
    te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return safe_div(float(active.mean() * TRADING_DAYS), te)


def beta_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    aligned = align_portfolio_and_benchmark(portfolio_returns, benchmark_returns)
    var_b = float(aligned["benchmark"].var())
    beta = np.nan if var_b == 0 else float(aligned["portfolio"].cov(aligned["benchmark"]) / var_b)
    alpha = float((aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()) * TRADING_DAYS) if not np.isnan(beta) else np.nan
    return beta, alpha


def relative_var_cvar_es(portfolio_returns: pd.Series, benchmark_returns: pd.Series, alpha: float = 0.95) -> dict:
    aligned = align_portfolio_and_benchmark(portfolio_returns, benchmark_returns)
    active = aligned["portfolio"] - aligned["benchmark"]
    q = float(active.quantile(1 - alpha))
    tail = active[active <= q]
    cvar = float(tail.mean()) if len(tail) else q
    return {"relative_var": q, "relative_cvar": cvar, "relative_es": cvar}
