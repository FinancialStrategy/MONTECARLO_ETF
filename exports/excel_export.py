from __future__ import annotations

import io
import pandas as pd


def build_excel_report(tables: dict[str, pd.DataFrame]) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output
