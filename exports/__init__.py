# exports/__init__.py

"""
Exports package initialization for generating institutional reports.

This module provides utilities for exporting analytics results to Excel
and PDF formats suitable for executive and institutional reporting.
"""

from .excel_export import build_excel_report
from .pdf_export import build_pdf_report

__all__ = [
    "build_excel_report",
    "build_pdf_report",
]
