# exports/pdf_export.py

"""
PDF export module for the Institutional Portfolio Analytics Platform.

This implementation is designed to be safe for Streamlit Cloud and fpdf2
when using standard core fonts such as Arial / Helvetica.

Key protections:
- sanitizes Unicode characters that commonly break FPDF core fonts
- safely handles None / NaN values
- avoids multi_cell crashes from unsupported characters
- returns an in-memory BytesIO object for st.download_button
"""

from __future__ import annotations

import io
import math
import unicodedata
from typing import Iterable

from fpdf import FPDF


# =========================================================
# Text sanitization helpers
# =========================================================
def _is_nan_like(value) -> bool:
    """
    Detect NaN-like values safely.
    """
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except Exception:
        return False


def _sanitize_pdf_text(text) -> str:
    """
    Convert text into a PDF-safe ASCII-compatible form for built-in FPDF fonts.

    Why needed:
    FPDF core fonts do not support many Unicode glyphs.
    Streamlit Cloud often surfaces this as an FPDFException inside multi_cell().

    This sanitizer:
    - converts None / NaN to readable placeholders
    - normalizes Unicode text
    - replaces common problematic glyphs
    - strips unsupported remaining characters
    """
    if _is_nan_like(text):
        return "N/A"

    if text is None:
        return "N/A"

    s = str(text)

    # Common replacements for problematic typography/symbols
    replacements = {
        "\u2013": "-",   # en dash
        "\u2014": "-",   # em dash
        "\u2212": "-",   # minus sign
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2022": "-",   # bullet
        "\u00b7": "-",   # middle dot
        "\u00a0": " ",   # non-breaking space
        "\u2026": "...", # ellipsis
        "\u20ac": "EUR", # euro
        "\u00a3": "GBP", # pound
        "\u00a5": "JPY", # yen
        "\u00ae": "(R)",
        "\u2122": "(TM)",
        "\u00b0": " deg ",
    }

    for bad, good in replacements.items():
        s = s.replace(bad, good)

    # Normalize accents and compatibility characters
    s = unicodedata.normalize("NFKD", s)

    # Keep only characters safe for latin-1/core font workflow
    s = s.encode("latin-1", "ignore").decode("latin-1")

    # Remove control chars except newline/tab
    cleaned_chars = []
    for ch in s:
        code = ord(ch)
        if ch in ("\n", "\t"):
            cleaned_chars.append(ch)
        elif 32 <= code <= 255:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")

    s = "".join(cleaned_chars)

    # Collapse overly repeated whitespace
    s = " ".join(s.split()) if "\n" not in s else "\n".join(" ".join(part.split()) for part in s.splitlines())

    if not s.strip():
        return "-"

    return s


def _safe_multicell(pdf: FPDF, width: float, line_height: float, text) -> None:
    """
    Write text safely using multi_cell after sanitization.
    """
    safe_text = _sanitize_pdf_text(text)

    # Preserve blank lines intentionally
    if safe_text.strip() == "":
        pdf.ln(line_height)
        return

    pdf.multi_cell(width, line_height, safe_text)


# =========================================================
# Main export function
# =========================================================
def build_pdf_report(summary_lines: Iterable[str]) -> io.BytesIO:
    """
    Build a simple multi-page institutional PDF report from a list of summary lines.

    Parameters
    ----------
    summary_lines : iterable of str
        Report text lines to render.

    Returns
    -------
    io.BytesIO
        In-memory PDF buffer for Streamlit download_button.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_title("Institutional Portfolio Analytics Report")
    pdf.set_author("OpenAI")
    pdf.set_creator("Institutional Portfolio Analytics Platform")

    # -----------------------------
    # Page 1 - Cover / Header
    # -----------------------------
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    _safe_multicell(pdf, 0, 10, "Institutional Portfolio Analytics Report")

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    _safe_multicell(
        pdf,
        0,
        7,
        "Executive summary generated from the Streamlit-based analytics platform."
    )

    pdf.ln(4)
    pdf.set_draw_color(120, 120, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # -----------------------------
    # Summary section
    # -----------------------------
    pdf.set_font("Helvetica", "B", 13)
    _safe_multicell(pdf, 0, 8, "Summary")

    pdf.ln(1)
    pdf.set_font("Helvetica", "", 11)

    if summary_lines is None:
        summary_lines = []

    for line in summary_lines:
        safe_line = _sanitize_pdf_text(line)

        # Add page break safety
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_font("Helvetica", "", 11)

        _safe_multicell(pdf, 0, 7, safe_line)

    # -----------------------------
    # Footer note
    # -----------------------------
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    _safe_multicell(
        pdf,
        0,
        5,
        "Note: This report is model-based and intended for analytical use only. "
        "It does not constitute investment advice."
    )

    # -----------------------------
    # Export as bytes
    # -----------------------------
    raw = pdf.output(dest="S")

    # fpdf2 may return str or bytes depending on version/environment
    if isinstance(raw, str):
        raw = raw.encode("latin-1", errors="ignore")

    buffer = io.BytesIO(raw)
    buffer.seek(0)
    return buffer
