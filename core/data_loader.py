# core/data_loader.py

"""
Robust market data loader for the Institutional Portfolio Analytics Platform.

Features:
- downloads Yahoo Finance data
- supports single and multi-ticker downloads
- auto-adjusted prices
- robust column extraction
- limited forward filling for sparse gaps
- data coverage checks
- minimum-history filtering
- separate benchmark handling
- simple/log return support
- deployment-safe error handling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class DataQualityConfig:
    """
    Configuration for price cleaning and validation.
    """
    min_history_rows: int = 60
    min_asset_coverage: float = 0.70
    max_forward_fill_days: int = 3
    benchmark_tickers: tuple = ("SPY", "QQQ")


class DataLoader:
    """
    Download and prepare price/return data for portfolio analytics.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date,
        end_date,
        use_log_returns: bool = False,
        quality_config: Optional[DataQualityConfig] = None,
    ):
        self.tickers = list(dict.fromkeys(tickers))
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.use_log_returns = use_log_returns
        self.quality_config = quality_config or DataQualityConfig()

    # =====================================================
    # Public API
    # =====================================================
    def fetch_prices(self) -> pd.DataFrame:
        """
        Download adjusted close prices from Yahoo Finance.

        Returns:
            DataFrame with one column per ticker.
        """
        all_tickers = list(dict.fromkeys(self.tickers + list(self.quality_config.benchmark_tickers)))

        raw = yf.download(
            tickers=all_tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )

        if raw is None or len(raw) == 0:
            raise ValueError("Yahoo Finance returned no data.")

        prices = self._extract_close_prices(raw, all_tickers)

        if prices.empty:
            raise ValueError("No usable close prices could be extracted from Yahoo Finance data.")

        prices = prices.sort_index()
        prices = prices[~prices.index.duplicated(keep="last")]

        return prices

    def compute_returns(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
        """
        Clean prices, validate data sufficiency, and compute returns.

        Returns dict with:
            - prices
            - returns
            - benchmark_prices
            - benchmark_returns
            - current_prices
            - valid_tickers
        """
        if prices is None or prices.empty:
            raise ValueError("Price DataFrame is empty before return computation.")

        prices = prices.copy().sort_index()

        # Separate portfolio assets and benchmark assets
        requested_assets = [t for t in self.tickers if t in prices.columns]
        benchmark_assets = [t for t in self.quality_config.benchmark_tickers if t in prices.columns]

        if len(requested_assets) == 0:
            raise ValueError("None of the selected portfolio tickers were found in the downloaded price data.")

        asset_prices_raw = prices[requested_assets].copy()
        benchmark_prices_raw = prices[benchmark_assets].copy() if benchmark_assets else pd.DataFrame(index=prices.index)

        # Clean portfolio asset prices robustly
        cleaned_asset_prices = self._clean_asset_price_matrix(asset_prices_raw)

        if cleaned_asset_prices.empty:
            raise ValueError("No usable portfolio price history remains after cleaning.")

        if cleaned_asset_prices.shape[1] == 0:
            raise ValueError("All selected portfolio assets were removed during cleaning due to poor data coverage.")

        if cleaned_asset_prices.shape[0] < self.quality_config.min_history_rows:
            raise ValueError("Insufficient aligned price history after cleaning.")

        # Clean benchmarks separately so they do not destroy the asset matrix
        cleaned_benchmark_prices = self._clean_benchmark_price_matrix(
            benchmark_prices_raw,
            asset_index=cleaned_asset_prices.index,
        )

        # Returns
        asset_returns = self._price_to_return(cleaned_asset_prices)

        if asset_returns.empty or asset_returns.shape[0] < max(30, self.quality_config.min_history_rows // 2):
            raise ValueError("Insufficient aligned return history after cleaning.")

        benchmark_returns = self._price_to_return(cleaned_benchmark_prices) if not cleaned_benchmark_prices.empty else pd.DataFrame()

        current_prices = cleaned_asset_prices.iloc[-1]
        valid_tickers = list(cleaned_asset_prices.columns)

        return {
            "prices": cleaned_asset_prices,
            "returns": asset_returns,
            "benchmark_prices": cleaned_benchmark_prices,
            "benchmark_returns": benchmark_returns,
            "current_prices": current_prices,
            "valid_tickers": valid_tickers,
        }

    # =====================================================
    # Internal helpers
    # =====================================================
    def _extract_close_prices(self, raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """
        Extract close prices robustly from yfinance download output.
        Handles both MultiIndex and flat-column formats.
        """
        series_map = {}

        if isinstance(raw.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    if ticker in raw.columns.get_level_values(0):
                        subcols = raw[ticker]
                        if "Close" in subcols.columns:
                            s = subcols["Close"].copy()
                        elif "Adj Close" in subcols.columns:
                            s = subcols["Adj Close"].copy()
                        else:
                            continue

                        s.name = ticker
                        series_map[ticker] = s
                except Exception:
                    continue
        else:
            # Single ticker fallback
            possible_close_cols = ["Close", "Adj Close"]
            for col in possible_close_cols:
                if col in raw.columns:
                    only_ticker = tickers[0]
                    s = raw[col].copy()
                    s.name = only_ticker
                    series_map[only_ticker] = s
                    break

        if not series_map:
            return pd.DataFrame()

        prices = pd.concat(series_map.values(), axis=1)
        return prices

    def _clean_asset_price_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly clean selected portfolio asset price matrix.
        """
        if prices.empty:
            return pd.DataFrame()

        df = prices.copy()

        # Ensure numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop fully empty rows first
        df = df.dropna(how="all")

        if df.empty:
            return pd.DataFrame()

        # Asset coverage filter
        coverage = df.notna().mean()
        keep_cols = coverage[coverage >= self.quality_config.min_asset_coverage].index.tolist()

        df = df[keep_cols]

        if df.empty:
            return pd.DataFrame()

        # Limited forward fill for short gaps only
        df = df.ffill(limit=self.quality_config.max_forward_fill_days)

        # Drop rows where all are missing after ffill
        df = df.dropna(how="all")

        if df.empty:
            return pd.DataFrame()

        # Final aligned matrix: require all kept assets present
        aligned = df.dropna(how="any")

        # If this is too strict and leaves too little data, try a second-pass relaxation
        if aligned.shape[0] < self.quality_config.min_history_rows:
            aligned = self._relaxed_alignment(df)

        # Final integrity checks
        aligned = aligned.sort_index()
        aligned = aligned[~aligned.index.duplicated(keep="last")]

        # Remove constant-price columns if any
        good_cols = []
        for col in aligned.columns:
            s = aligned[col].dropna()
            if len(s) >= self.quality_config.min_history_rows and s.nunique() > 1:
                good_cols.append(col)

        aligned = aligned[good_cols]

        # After dropping bad columns, align again
        if not aligned.empty:
            aligned = aligned.dropna(how="any")

        return aligned

    def _relaxed_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Second-pass alignment when strict intersection leaves too few rows.

        Strategy:
        - rank columns by coverage
        - iteratively keep the best-covered assets until enough common history remains
        """
        if df.empty:
            return pd.DataFrame()

        coverage = df.notna().mean().sort_values(ascending=False)
        ranked_cols = coverage.index.tolist()

        best_candidate = pd.DataFrame()

        for k in range(len(ranked_cols), 0, -1):
            subset_cols = ranked_cols[:k]
            candidate = df[subset_cols].dropna(how="any")

            if candidate.shape[0] >= self.quality_config.min_history_rows:
                return candidate

            if candidate.shape[0] > best_candidate.shape[0]:
                best_candidate = candidate

        return best_candidate

    def _clean_benchmark_price_matrix(
        self,
        benchmark_prices: pd.DataFrame,
        asset_index: pd.Index,
    ) -> pd.DataFrame:
        """
        Clean benchmark prices without forcing them to destroy portfolio alignment.
        Benchmarks are reindexed to the asset matrix index.
        """
        if benchmark_prices is None or benchmark_prices.empty:
            return pd.DataFrame(index=asset_index)

        df = benchmark_prices.copy()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Reindex to asset dates
        df = df.reindex(asset_index)

        # Limited fill
        df = df.ffill(limit=self.quality_config.max_forward_fill_days)

        # Keep only benchmarks with sufficient coverage on aligned dates
        coverage = df.notna().mean()
        keep_cols = coverage[coverage >= 0.80].index.tolist()
        df = df[keep_cols] if keep_cols else pd.DataFrame(index=asset_index)

        # Final cleanup
        if not df.empty:
            # Keep row-wise NaNs; they will be handled later by pairwise alignment in relative analytics
            df = df.dropna(how="all")

        return df

    def _price_to_return(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Convert prices to returns using either simple or log-return mode.
        Output is always simple-return space for downstream consistency.
        """
        if prices is None or prices.empty:
            return pd.DataFrame()

        if self.use_log_returns:
            log_ret = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna(how="all")
            ret = np.exp(log_ret) - 1.0
        else:
            ret = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")

        # Final cleanup: drop columns with no variability
        keep_cols = []
        for col in ret.columns:
            s = ret[col].dropna()
            if len(s) > 1 and s.nunique() > 1:
                keep_cols.append(col)

        ret = ret[keep_cols] if keep_cols else pd.DataFrame(index=ret.index)

        # For the portfolio return matrix we generally want full alignment across kept assets
        if not ret.empty:
            ret = ret.dropna(how="any")

        return ret
