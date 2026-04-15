from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(a: float, b: float) -> float:
    if b is None:
        return np.nan
    try:
        if np.isnan(b) or b == 0:
            return np.nan
    except Exception:
        if b == 0:
            return np.nan
    return a / b


def nearest_psd_cov(cov: pd.DataFrame) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, 1e-10, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    repaired = (repaired + repaired.T) / 2.0
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)


def drawdown_series(values: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(values, dtype=float)
    running_max = s.cummax()
    return s / running_max - 1.0


def max_drawdown_from_values(values: pd.Series | np.ndarray) -> float:
    return float(drawdown_series(values).min())


def annualize_simple_mean_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return np.nan
    return float(returns.mean() * periods_per_year)
