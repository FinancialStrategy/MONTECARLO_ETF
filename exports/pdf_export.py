# exports/pdf_export.py

import io
from fpdf import FPDF

def build_pdf_report(summary_lines: list[str]):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Institutional Portfolio Analytics Report", ln=True)

    pdf.set_font("Arial", size=11)
    pdf.ln(4)
    for line in summary_lines:
        pdf.multi_cell(0, 7, line)

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        raw = raw.encode("latin-1")
    return io.BytesIO(raw)
