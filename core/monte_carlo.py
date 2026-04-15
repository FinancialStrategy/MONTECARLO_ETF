from __future__ import annotations

import numpy as np
import pandas as pd

from core.utils import nearest_psd_cov


class MonteCarloEngine:
    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame, num_simulations: int = 10000, forecast_days: int = 252):
        self.mean_returns = mean_returns
        self.cov_matrix = nearest_psd_cov(cov_matrix)
        self.num_simulations = num_simulations
        self.forecast_days = forecast_days

    def run(self, weights: np.ndarray, initial_investment: float) -> dict:
        weights = np.asarray(weights, dtype=float)
        mu = self.mean_returns.values
        cov = self.cov_matrix.values

        chol = np.linalg.cholesky(cov)
        z = np.random.normal(size=(self.num_simulations, self.forecast_days, len(weights)))
        correlated_noise = np.einsum("sda,ab->sdb", z, chol.T)
        simulated_asset_returns = correlated_noise + mu
        simulated_portfolio_returns = np.einsum("sda,a->sd", simulated_asset_returns, weights)

        portfolio_values = np.zeros((self.num_simulations, self.forecast_days + 1))
        portfolio_values[:, 0] = initial_investment
        portfolio_values[:, 1:] = initial_investment * np.cumprod(1 + simulated_portfolio_returns, axis=1)

        final_values = portfolio_values[:, -1]
        cumulative_max = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns = portfolio_values / cumulative_max - 1.0
        max_drawdowns = drawdowns.min(axis=1)

        return {
            "portfolio_values": portfolio_values,
            "daily_returns": simulated_portfolio_returns,
            "final_values": final_values,
            "max_drawdowns": max_drawdowns,
            "expected_value": float(final_values.mean()),
            "median_value": float(np.median(final_values)),
            "std_value": float(final_values.std(ddof=1)),
        }
