# core/optimization.py

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
    def __init__(self, returns: pd.DataFrame, risk_free_rate=DEFAULT_RISK_FREE_RATE, covariance_method="Sample"):
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self._build_covariance(covariance_method)

    def _build_covariance(self, method):
        if method == "Ledoit-Wolf" and SKLEARN_AVAILABLE:
            lw = LedoitWolf()
            lw.fit(self.returns.values)
            cov = pd.DataFrame(lw.covariance_, index=self.returns.columns, columns=self.returns.columns)
        else:
            cov = self.returns.cov()
        return nearest_psd_cov(cov)

    def portfolio_stats(self, weights):
        w = np.array(weights, dtype=float)
        mu_daily = np.dot(self.mean_returns.values, w)
        ann_ret = (1 + mu_daily) ** TRADING_DAYS - 1
        ann_vol = np.sqrt(w.T @ (self.cov_matrix.values * TRADING_DAYS) @ w)
        sharpe = safe_div(ann_ret - self.risk_free_rate, ann_vol)
        return ann_ret, ann_vol, sharpe

    def optimize(self, objective="max_sharpe"):
        n = len(self.returns.columns)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def min_vol(w):
            return self.portfolio_stats(w)[1]

        def neg_sharpe(w):
            shr = self.portfolio_stats(w)[2]
            return 1e6 if np.isnan(shr) else -shr

        def neg_ret(w):
            return -self.portfolio_stats(w)[0]

        func = {
            "min_volatility": min_vol,
            "max_sharpe": neg_sharpe,
            "max_return": neg_ret,
        }[objective]

        res = minimize(func, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else x0

    def optimize_tracking_error(self, benchmark_returns: pd.Series):
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
            te = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
            return te

        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        return res.x if res.success else x0
