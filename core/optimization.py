# core/optimization.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _ensure_unique_sorted_index(obj: pd.DataFrame | pd.Series):
    obj = obj.copy()
    obj = obj[~obj.index.duplicated(keep="last")]
    obj = obj.sort_index()
    return obj


def _safe_divide(a: float, b: float) -> float:
    try:
        if b is None or np.isnan(b) or b == 0:
            return np.nan
    except Exception:
        return np.nan
    return a / b


def _nearest_psd_cov(cov: pd.DataFrame) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh(cov.values)
    vals = np.clip(vals, 1e-10, None)
    repaired = vecs @ np.diag(vals) @ vecs.T
    repaired = (repaired + repaired.T) / 2.0
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)


class PortfolioOptimizer:
    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.03,
        covariance_method: str = "Sample",
    ):
        if returns is None or len(returns) == 0:
            raise ValueError("Returns DataFrame is empty.")

        if not isinstance(returns, pd.DataFrame):
            raise ValueError("PortfolioOptimizer expects returns as a pandas DataFrame.")

        cleaned = returns.copy()
        for col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = _ensure_unique_sorted_index(cleaned)
        cleaned = cleaned.dropna(how="any")

        if cleaned.empty:
            raise ValueError("Returns DataFrame became empty after cleaning.")

        if cleaned.shape[1] == 0:
            raise ValueError("Returns DataFrame contains no valid assets after cleaning.")

        self.returns = cleaned
        self.risk_free_rate = risk_free_rate
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self._build_covariance(covariance_method)

    def _build_covariance(self, method: str) -> pd.DataFrame:
        method = (method or "Sample").strip()

        if method == "Ledoit-Wolf" and SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov = pd.DataFrame(
                lw.covariance_,
                index=self.returns.columns,
                columns=self.returns.columns,
            )
        else:
            cov = self.returns.cov()

        return _nearest_psd_cov(cov)

    def portfolio_stats(self, weights: np.ndarray):
        w = np.asarray(weights, dtype=float)
        mu_daily = float(np.dot(self.mean_returns.values, w))
        ann_return = (1.0 + mu_daily) ** TRADING_DAYS - 1.0
        ann_vol = float(np.sqrt(w.T @ (self.cov_matrix.values * TRADING_DAYS) @ w))
        sharpe = _safe_divide(ann_return - self.risk_free_rate, ann_vol)
        return ann_return, ann_vol, sharpe

    def optimize(self, objective: str = "max_sharpe") -> np.ndarray:
        n = len(self.returns.columns)
        x0 = np.repeat(1.0 / n, n)

        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def min_vol(w):
            return self.portfolio_stats(w)[1]

        def neg_sharpe(w):
            shr = self.portfolio_stats(w)[2]
            return 1e6 if np.isnan(shr) else -shr

        def neg_return(w):
            return -self.portfolio_stats(w)[0]

        objective_map = {
            "min_volatility": min_vol,
            "max_sharpe": neg_sharpe,
            "max_return": neg_return,
        }

        if objective not in objective_map:
            raise ValueError(f"Unsupported optimization objective: {objective}")

        res = minimize(
            objective_map[objective],
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success and res.x is not None:
            return np.asarray(res.x, dtype=float)

        return x0

    def optimize_tracking_error(self, benchmark_returns: pd.Series | pd.DataFrame) -> np.ndarray:
        if benchmark_returns is None or len(benchmark_returns) == 0:
            raise ValueError("Benchmark returns are empty.")

        asset_returns = self.returns.copy()
        asset_returns = _ensure_unique_sorted_index(asset_returns)
        asset_returns = asset_returns.dropna(how="any")

        if asset_returns.empty:
            raise ValueError("Asset returns are empty after cleaning in optimize_tracking_error.")

        if isinstance(benchmark_returns, pd.DataFrame):
            if benchmark_returns.shape[1] == 0:
                raise ValueError("Benchmark DataFrame contains no columns.")
            benchmark = benchmark_returns.iloc[:, 0].copy()
        else:
            benchmark = benchmark_returns.copy()

        benchmark = pd.to_numeric(benchmark, errors="coerce")
        benchmark = benchmark.replace([np.inf, -np.inf], np.nan)
        benchmark = _ensure_unique_sorted_index(benchmark)
        benchmark = benchmark.dropna()

        if benchmark.empty:
            raise ValueError("Benchmark returns are empty after cleaning in optimize_tracking_error.")

        combined = pd.concat([asset_returns, benchmark.rename("benchmark")], axis=1, join="inner")
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined = combined.dropna(how="any")

        if combined.empty:
            raise ValueError("No overlapping history between asset returns and benchmark returns.")

        benchmark_aligned = combined["benchmark"].copy()
        asset_aligned = combined.drop(columns=["benchmark"]).copy()

        if asset_aligned.empty:
            raise ValueError("No asset return columns remain after benchmark alignment.")

        if asset_aligned.shape[0] < 20:
            raise ValueError("Insufficient overlapping history for tracking error optimization.")

        X = asset_aligned.values
        b = benchmark_aligned.values.reshape(-1)

        n_assets = X.shape[1]
        x0 = np.repeat(1.0 / n_assets, n_assets)

        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            w = np.asarray(w, dtype=float)
            portfolio_path = X @ w
            active = portfolio_path - b
            te = np.std(active, ddof=1) * np.sqrt(TRADING_DAYS)

            if np.isnan(te) or np.isinf(te):
                return 1e6
            return float(te)

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if res.success and res.x is not None:
            return np.asarray(res.x, dtype=float)

        return x0
