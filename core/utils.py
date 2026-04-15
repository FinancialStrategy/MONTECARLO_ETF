# core/utils.py

import numpy as np
import pandas as pd

def safe_div(a, b):
    if b is None:
        return np.nan
    try:
        if np.isnan(b) or b == 0:
            return np.nan
    except Exception:
        pass
    return a / b

def nearest_psd_cov(cov: pd.DataFrame) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, 1e-10, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    repaired = (repaired + repaired.T) / 2.0
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)

def max_drawdown_from_values(values):
    values = np.asarray(values, dtype=float)
    running_max = np.maximum.accumulate(values)
    dd = values / running_max - 1.0
    return dd.min()

def drawdown_series(values):
    values = pd.Series(values).astype(float)
    running_max = values.cummax()
    return values / running_max - 1.0
