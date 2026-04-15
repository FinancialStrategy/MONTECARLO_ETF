from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_TAU


class BlackLittermanModel:
    def __init__(self, cov_matrix: pd.DataFrame, market_weights: np.ndarray, risk_aversion: float = 2.5, tau: float = DEFAULT_TAU):
        self.cov = cov_matrix.copy()
        self.market_weights = np.asarray(market_weights, dtype=float)
        self.risk_aversion = risk_aversion
        self.tau = tau

    def equilibrium_returns(self) -> pd.Series:
        pi = self.risk_aversion * self.cov.values @ self.market_weights
        return pd.Series(pi, index=self.cov.index, name="Pi")

    def build_omega(self, P: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        base = np.diag(np.diag(P @ (self.tau * self.cov.values) @ P.T))
        conf = np.clip(confidences, 1e-4, 0.9999)
        scale = (1.0 - conf) / conf
        return np.diag(np.diag(base) * scale)

    def posterior(self, P: np.ndarray, Q: np.ndarray, confidences: np.ndarray) -> dict:
        Sigma = self.cov.values
        Pi = self.equilibrium_returns().values.reshape(-1, 1)
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float).reshape(-1, 1)
        Omega = self.build_omega(P, np.asarray(confidences, dtype=float))

        tauSigma_inv = np.linalg.inv(self.tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)
        middle = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
        mu_bl = middle @ (tauSigma_inv @ Pi + P.T @ Omega_inv @ Q)
        sigma_bl = Sigma + middle

        return {
            "equilibrium_returns": self.equilibrium_returns(),
            "posterior_returns": pd.Series(mu_bl.flatten(), index=self.cov.index),
            "posterior_cov": pd.DataFrame(sigma_bl, index=self.cov.index, columns=self.cov.columns),
        }
