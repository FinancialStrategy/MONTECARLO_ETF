# core/__init__.py

"""
Core package for the Institutional Portfolio Analytics Platform.

This file is intentionally minimal to avoid import-chain failures during
package initialization on deployment environments such as Streamlit Cloud.

All submodules should be imported directly from app.py, for example:

    from core.data_loader import DataLoader
    from core.risk import summary_risk_table
    from core.optimization import PortfolioOptimizer
    from core.monte_carlo import MonteCarloEngine
    from core.black_litterman import BlackLittermanModel
    from core.regime import RegimeDetector
    from core.relative_risk import tracking_error, information_ratio, beta_alpha, relative_var_cvar_es
    from core.reporting import allocation_table, benchmark_probability_table, percentile_table
"""
