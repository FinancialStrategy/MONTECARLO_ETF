from __future__ import annotations

import io
import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF


def dataframe_to_png(df: pd.DataFrame, title: str, output_path: str) -> str:
    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.45 * (len(df) + 1))))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_pdf_report(summary_lines: Iterable[str], image_paths: list[str] | None = None) -> io.BytesIO:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Institutional Portfolio Analytics Report", ln=True)
    pdf.set_font("Helvetica", size=11)
    pdf.ln(2)
    for line in summary_lines:
        pdf.multi_cell(0, 7, str(line))

    for image_path in image_paths or []:
        if os.path.exists(image_path):
            pdf.add_page()
            pdf.image(image_path, x=10, y=15, w=190)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    return io.BytesIO(raw)
