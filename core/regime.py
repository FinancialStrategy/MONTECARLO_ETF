from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    def __init__(self, returns: pd.Series, window: int = 63):
        self.returns = pd.Series(returns).dropna()
        self.window = window

    def detect(self) -> pd.DataFrame:
        rolling_ret = self.returns.rolling(self.window).mean() * 252
        rolling_vol = self.returns.rolling(self.window).std() * np.sqrt(252)
        vol_q1 = rolling_vol.quantile(0.33)
        vol_q2 = rolling_vol.quantile(0.66)

        regime = pd.Series(index=self.returns.index, dtype="object")
        for idx in regime.index:
            r = rolling_ret.loc[idx]
            v = rolling_vol.loc[idx]
            if pd.isna(r) or pd.isna(v):
                regime.loc[idx] = None
            elif r > 0 and v <= vol_q1:
                regime.loc[idx] = "Bull / Low Vol"
            elif r < 0 and v >= vol_q2:
                regime.loc[idx] = "Bear / High Vol"
            else:
                regime.loc[idx] = "Neutral"

        return pd.DataFrame({"return_ann": rolling_ret, "vol_ann": rolling_vol, "regime": regime}).dropna()
