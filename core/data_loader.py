# core/data_loader.py

import pandas as pd
import numpy as np
import yfinance as yf

class DataLoader:
    def __init__(self, tickers, start_date, end_date, use_log_returns=False):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.use_log_returns = use_log_returns

    def fetch_prices(self):
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

        prices = pd.concat(series_map.values(), axis=1).sort_index()
        prices = prices.dropna(how="all")
        return prices

    def compute_returns(self, prices: pd.DataFrame):
        selected = prices[self.tickers].dropna()
        benchmarks = prices[[c for c in ["SPY", "QQQ"] if c in prices.columns]].dropna()

        if self.use_log_returns:
            rets = np.log(selected / selected.shift(1)).dropna()
            bench = np.log(benchmarks / benchmarks.shift(1)).dropna()
            rets = np.exp(rets) - 1
            bench = np.exp(bench) - 1
        else:
            rets = selected.pct_change().dropna()
            bench = benchmarks.pct_change().dropna()

        return {
            "prices": selected,
            "returns": rets,
            "benchmark_prices": benchmarks,
            "benchmark_returns": bench,
            "current_prices": selected.iloc[-1]
        }
