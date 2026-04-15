from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


class DataLoader:
    def __init__(self, tickers, start_date, end_date, use_log_returns=False):
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.use_log_returns = use_log_returns

    def fetch_prices(self) -> pd.DataFrame:
        tickers = list(dict.fromkeys(self.tickers + ["SPY", "QQQ"]))
        raw = yf.download(
            tickers=tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        series_map = {}
        for t in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw[t]["Close"].copy()
                else:
                    s = raw["Close"].copy()
                s.name = t
                series_map[t] = s
            except Exception:
                continue

        if not series_map:
            raise ValueError("No prices downloaded from Yahoo Finance.")

        prices = pd.concat(series_map.values(), axis=1).sort_index().dropna(how="all")
        return prices

    def compute_returns(self, prices: pd.DataFrame) -> dict:
        selected = prices[self.tickers].dropna()
        benchmark_cols = [c for c in ["SPY", "QQQ"] if c in prices.columns]
        benchmarks = prices[benchmark_cols].dropna() if benchmark_cols else pd.DataFrame(index=prices.index)

        if selected.shape[0] < 60:
            raise ValueError("Insufficient aligned price history after cleaning.")

        if self.use_log_returns:
            returns = np.log(selected / selected.shift(1)).dropna()
            returns = np.exp(returns) - 1
            if not benchmarks.empty:
                benchmark_returns = np.log(benchmarks / benchmarks.shift(1)).dropna()
                benchmark_returns = np.exp(benchmark_returns) - 1
            else:
                benchmark_returns = pd.DataFrame(index=returns.index)
        else:
            returns = selected.pct_change().dropna()
            benchmark_returns = benchmarks.pct_change().dropna() if not benchmarks.empty else pd.DataFrame(index=returns.index)

        return {
            "prices": selected,
            "returns": returns,
            "benchmark_prices": benchmarks,
            "benchmark_returns": benchmark_returns,
            "current_prices": selected.iloc[-1],
        }
