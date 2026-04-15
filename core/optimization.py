from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import TRADING_DAYS, DEFAULT_RISK_FREE_RATE
from core.utils import nearest_psd_cov, safe_div

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = DEFAULT_RISK_FREE_RATE, covariance_method: str = "Sample"):
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self._build_covariance(covariance_method)

    def _build_covariance(self, method: str) -> pd.DataFrame:
        if method == "Ledoit-Wolf" and SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov = pd.DataFrame(lw.covariance_, index=self.returns.columns, columns=self.returns.columns)
        else:
            cov = self.returns.cov()
        return nearest_psd_cov(cov)

    def portfolio_stats(self, weights: np.ndarray, mean_returns: pd.Series | None = None, cov_matrix: pd.DataFrame | None = None) -> tuple[float, float, float]:
        mean_returns = self.mean_returns if mean_returns is None else mean_returns
        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix
        w = np.asarray(weights, dtype=float)
        mu_daily = float(np.dot(mean_returns.values, w))
        ann_ret = (1 + mu_daily) ** TRADING_DAYS - 1
        ann_vol = float(np.sqrt(w.T @ (cov_matrix.values * TRADING_DAYS) @ w))
        sharpe = safe_div(ann_ret - self.risk_free_rate, ann_vol)
        return ann_ret, ann_vol, sharpe

    def optimize(self, objective: str = "max_sharpe", mean_returns: pd.Series | None = None, cov_matrix: pd.DataFrame | None = None) -> np.ndarray:
        mean_returns = self.mean_returns if mean_returns is None else mean_returns
        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix
        n = len(self.returns.columns)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def min_vol(w):
            return self.portfolio_stats(w, mean_returns, cov_matrix)[1]

        def neg_sharpe(w):
            shr = self.portfolio_stats(w, mean_returns, cov_matrix)[2]
            return 1e6 if np.isnan(shr) else -shr

        def neg_ret(w):
            return -self.portfolio_stats(w, mean_returns, cov_matrix)[0]

        objective_map = {
            "min_volatility": min_vol,
            "max_sharpe": neg_sharpe,
            "max_return": neg_ret,
        }
        res = minimize(objective_map[objective], x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else x0

    def optimize_tracking_error(self, benchmark_returns: pd.Series) -> np.ndarray:
        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        asset_returns = aligned[self.returns.columns]
        benchmark = aligned[benchmark_returns.name]

        n = len(asset_returns.columns)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            port = asset_returns @ w
            active = port - benchmark
            return float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else x0

    def optimize_active_risk_budget(self, benchmark_returns: pd.Series, target_tracking_error: float = 0.05, penalty: float = 10.0) -> np.ndarray:
        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        asset_returns = aligned[self.returns.columns]
        benchmark = aligned[benchmark_returns.name]
        n = len(asset_returns.columns)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        mu = asset_returns.mean() * TRADING_DAYS

        def objective(w):
            port = asset_returns @ w
            active = port - benchmark
            te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
            active_return = (port.mean() - benchmark.mean()) * TRADING_DAYS
            return float(-(active_return) + penalty * (te - target_tracking_error) ** 2)

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else x0

    def benchmark_relative_frontier(self, benchmark_returns: pd.Series, points: int = 100) -> pd.DataFrame:
        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        asset_returns = aligned[self.returns.columns]
        benchmark = aligned[benchmark_returns.name]
        out = []
        for _ in range(points):
            w = np.random.random(asset_returns.shape[1])
            w /= w.sum()
            port = asset_returns @ w
            active = port - benchmark
            te = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
            active_ret = float(active.mean() * TRADING_DAYS)
            ir = safe_div(active_ret, te)
            out.append({"active_return": active_ret * 100, "tracking_error": te * 100, "information_ratio": ir, "weights": w})
        return pd.DataFrame(out)
