# core/reporting.py

import pandas as pd
import numpy as np
from config import BENCHMARK_ASSUMPTIONS

def allocation_table(tickers, weights, universe):
    df = pd.DataFrame({
        "Ticker": tickers,
        "Name": [universe[t]["name"] for t in tickers],
        "Category": [universe[t]["category"] for t in tickers],
        "Expense Ratio (%)": [universe[t]["expense_ratio"] for t in tickers],
        "Weight (%)": np.array(weights) * 100
    })
    df["Weighted Expense Contribution (%)"] = df["Expense Ratio (%)"] * df["Weight (%)"] / 100
    return df.sort_values("Weight (%)", ascending=False).reset_index(drop=True)

def benchmark_probability_table(final_values, initial_investment):
    rows = []
    for name, ann_ret in BENCHMARK_ASSUMPTIONS.items():
        final_benchmark = initial_investment * (1 + ann_ret)
        prob = (final_values > final_benchmark).mean() * 100
        rows.append({
            "Benchmark": name,
            "Assumed Annual Return (%)": ann_ret * 100,
            "Benchmark Final Value ($)": final_benchmark,
            "Probability Outperforming (%)": prob
        })
    return pd.DataFrame(rows)

def percentile_table(final_values, initial_investment):
    p = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    values = np.percentile(final_values, p)
    returns = (values / initial_investment - 1) * 100
    return pd.DataFrame({
        "Percentile": p,
        "Terminal Value ($)": values,
        "Return (%)": returns
    })
