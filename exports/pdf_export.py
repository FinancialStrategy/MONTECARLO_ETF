# exports/pdf_export.py

"""
Robust PDF export module for the Institutional Portfolio Analytics Platform.

This version is hardened for Streamlit Cloud and fpdf2:
- sanitizes problematic Unicode
- wraps long tokens aggressively
- avoids width=0 in multi_cell
- falls back safely when multi_cell fails
- returns BytesIO for Streamlit download_button
"""

from __future__ import annotations

import io
import math
import re
import textwrap
import unicodedata
from typing import Iterable, List

from fpdf import FPDF
from fpdf.errors import FPDFException


# =========================================================
# Constants
# =========================================================
PAGE_WIDTH_MM = 210
LEFT_MARGIN_MM = 10
RIGHT_MARGIN_MM = 10
TOP_MARGIN_MM = 15
BOTTOM_MARGIN_MM = 15
USABLE_WIDTH_MM = PAGE_WIDTH_MM - LEFT_MARGIN_MM - RIGHT_MARGIN_MM


# =========================================================
# Utility helpers
# =========================================================
def _is_nan_like(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except Exception:
        return False


def _sanitize_pdf_text(text) -> str:
    """
    Convert text into a conservative Latin-1-safe representation.
    """
    if _is_nan_like(text):
        return "N/A"

    if text is None:
        return "N/A"

    s = str(text)

    replacements = {
        "\u2013": "-",     # en dash
        "\u2014": "-",     # em dash
        "\u2212": "-",     # minus sign
        "\u2018": "'",     # left single quote
        "\u2019": "'",     # right single quote
        "\u201c": '"',     # left double quote
        "\u201d": '"',     # right double quote
        "\u2022": "-",     # bullet
        "\u00b7": "-",     # middle dot
        "\u2026": "...",   # ellipsis
        "\u00a0": " ",     # nbsp
        "\u2009": " ",     # thin space
        "\u2002": " ",
        "\u2003": " ",
        "\u200b": "",      # zero width space
        "\ufeff": "",      # BOM
        "\t": "    ",
        "\r": "\n",
        "\u20ac": "EUR",
        "\u00a3": "GBP",
        "\u00a5": "JPY",
        "\u2122": "(TM)",
        "\u00ae": "(R)",
        "\u00a9": "(C)",
    }

    for bad, good in replacements.items():
        s = s.replace(bad, good)

    s = unicodedata.normalize("NFKD", s)
    s = s.encode("latin-1", "ignore").decode("latin-1")

    cleaned = []
    for ch in s:
        code = ord(ch)
        if ch == "\n":
            cleaned.append(ch)
        elif 32 <= code <= 255:
            cleaned.append(ch)
        else:
            cleaned.append(" ")

    s = "".join(cleaned)

    # Normalize whitespace line by line
    lines = []
    for line in s.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)

    s = "\n".join(lines).strip()

    return s if s else "-"


def _break_long_token(token: str, max_chunk: int = 40) -> List[str]:
    """
    Break a single long token into smaller chunks so PDF line wrapping won't fail.
    """
    if len(token) <= max_chunk:
        return [token]

    return [token[i:i + max_chunk] for i in range(0, len(token), max_chunk)]


def _wrap_text_for_pdf(text: str, max_chars_per_line: int = 95, max_token_len: int = 40) -> List[str]:
    """
    Wrap text conservatively for PDF output.

    Strategy:
    - split by original lines
    - split long tokens
    - rebuild text
    - wrap to safe line width
    """
    if not text:
        return ["-"]

    out_lines: List[str] = []

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()

        if raw_line == "":
            out_lines.append("")
            continue

        tokens = raw_line.split(" ")
        rebuilt_tokens: List[str] = []

        for token in tokens:
            if token == "":
                continue

            # Break URLs / huge identifiers / long words
            if len(token) > max_token_len:
                rebuilt_tokens.extend(_break_long_token(token, max_chunk=max_token_len))
            else:
                rebuilt_tokens.append(token)

        rebuilt_line = " ".join(rebuilt_tokens)

        wrapped = textwrap.wrap(
            rebuilt_line,
            width=max_chars_per_line,
            break_long_words=True,
            break_on_hyphens=True,
            replace_whitespace=False,
            drop_whitespace=True,
        )

        if not wrapped:
            out_lines.append("")
        else:
            out_lines.extend(wrapped)

    return out_lines if out_lines else ["-"]


def _safe_write_lines(pdf: FPDF, lines: List[str], line_height: float = 7.0) -> None:
    """
    Safely write pre-wrapped lines to PDF with a multi_cell fallback.
    """
    for line in lines:
        # preserve paragraph spacing
        if line.strip() == "":
            pdf.ln(line_height)
            continue

        # Add page if needed
        if pdf.get_y() > (297 - BOTTOM_MARGIN_MM - 10):
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)

        try:
            # Use fixed width instead of width=0
            pdf.multi_cell(USABLE_WIDTH_MM, line_height, line)
        except FPDFException:
            # Last-resort fallback: if multi_cell still fails, hard-split further
            tiny_chunks = textwrap.wrap(
                line,
                width=50,
                break_long_words=True,
                break_on_hyphens=True,
            )

            if not tiny_chunks:
                tiny_chunks = ["-"]

            for chunk in tiny_chunks:
                if pdf.get_y() > (297 - BOTTOM_MARGIN_MM - 10):
                    pdf.add_page()
                    pdf.set_font("Helvetica", size=11)

                # cell is even safer than multi_cell for short pieces
                safe_chunk = chunk[:200] if chunk else "-"
                try:
                    pdf.cell(USABLE_WIDTH_MM, line_height, safe_chunk, ln=1)
                except FPDFException:
                    # absolute emergency fallback
                    pdf.cell(USABLE_WIDTH_MM, line_height, "-", ln=1)


def _write_block(pdf: FPDF, text, line_height: float = 7.0, max_chars_per_line: int = 95) -> None:
    """
    End-to-end safe text writer for PDF blocks.
    """
    safe_text = _sanitize_pdf_text(text)
    wrapped_lines = _wrap_text_for_pdf(
        safe_text,
        max_chars_per_line=max_chars_per_line,
        max_token_len=40,
    )
    _safe_write_lines(pdf, wrapped_lines, line_height=line_height)


# =========================================================
# Main PDF builder
# =========================================================
def build_pdf_report(summary_lines: Iterable[str]) -> io.BytesIO:
    """
    Build a robust PDF report from summary lines.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(LEFT_MARGIN_MM, TOP_MARGIN_MM, RIGHT_MARGIN_MM)
    pdf.set_title("Institutional Portfolio Analytics Report")
    pdf.set_author("OpenAI")
    pdf.set_creator("Institutional Portfolio Analytics Platform")

    # -----------------------------------------------------
    # Page 1
    # -----------------------------------------------------
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    _write_block(pdf, "Institutional Portfolio Analytics Report", line_height=9, max_chars_per_line=70)

    pdf.ln(1)

    pdf.set_font("Helvetica", "", 11)
    _write_block(
        pdf,
        "Executive summary generated from the Streamlit-based analytics platform.",
        line_height=7,
        max_chars_per_line=90,
    )

    pdf.ln(3)

    # Divider
    y = pdf.get_y()
    pdf.set_draw_color(120, 120, 120)
    pdf.line(LEFT_MARGIN_MM, y, PAGE_WIDTH_MM - RIGHT_MARGIN_MM, y)
    pdf.ln(5)

    # Section title
    pdf.set_font("Helvetica", "B", 13)
    _write_block(pdf, "Summary", line_height=8, max_chars_per_line=80)
    pdf.ln(1)

    # Summary body
    pdf.set_font("Helvetica", "", 11)

    if summary_lines is None:
        summary_lines = []

    for line in summary_lines:
        _write_block(pdf, line, line_height=7, max_chars_per_line=95)

    pdf.ln(3)

    # Footer note
    pdf.set_font("Helvetica", "I", 9)
    _write_block(
        pdf,
        "Note: This report is model-based and intended for analytical use only. "
        "It does not constitute investment advice.",
        line_height=5,
        max_chars_per_line=100,
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    raw = pdf.output(dest="S")

    if isinstance(raw, str):
        raw = raw.encode("latin-1", errors="ignore")

    buffer = io.BytesIO(raw)
    buffer.seek(0)
    return buffer
