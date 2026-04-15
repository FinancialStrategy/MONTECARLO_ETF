# ui/__init__.py

"""
UI package initialization for the Institutional Portfolio Analytics Platform.

This module exposes commonly used UI components such as sidebar controls
and charting utilities for streamlined imports across the application.
"""

from .sidebar import render_sidebar
from .charts import (
    weight_bar_chart,
    category_pie_chart,
    monte_carlo_paths_chart,
    terminal_distribution_chart,
    regime_chart,
)

__all__ = [
    "render_sidebar",
    "weight_bar_chart",
    "category_pie_chart",
    "monte_carlo_paths_chart",
    "terminal_distribution_chart",
    "regime_chart",
]
