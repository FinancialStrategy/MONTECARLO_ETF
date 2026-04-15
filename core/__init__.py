# core/__init__.py

"""
Core analytics package for the Institutional Portfolio Analytics Platform.

This module aggregates the primary financial engines, including data loading,
risk analytics, optimization, Monte Carlo simulation, Black-Litterman modeling,
regime detection, and reporting utilities.
"""

from .data_loader import DataLoader
from .risk import summary_risk_table
from .optimization import PortfolioOptimizer
from .monte_carlo import MonteCarloEngine
from .black_litterman import BlackLittermanModel
from .regime import RegimeDetector
from .relative_risk import (
    tracking_error,
    information_ratio,
    beta_alpha,
    relative_var_cvar_es,
)
from .reporting import (
    allocation_table,
    benchmark_probability_table,
    percentile_table,
)

__all__ = [
    "DataLoader",
    "summary_risk_table",
    "PortfolioOptimizer",
    "MonteCarloEngine",
    "BlackLittermanModel",
    "RegimeDetector",
    "tracking_error",
    "information_ratio",
    "beta_alpha",
    "relative_var_cvar_es",
    "allocation_table",
    "benchmark_probability_table",
    "percentile_table",
]
