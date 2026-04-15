# core/regime.py

import pandas as pd
import numpy as np

class RegimeDetector:
    """
    Simple volatility-regime detector:
    bull / neutral / bear based on rolling returns and rolling volatility.
    """
    def __init__(self, returns: pd.Series, window: int = 63):
        self.returns = pd.Series(returns).dropna()
        self.window = window

    def detect(self):
        rolling_ret = self.returns.rolling(self.window).mean() * 252
        rolling_vol = self.returns.rolling(self.window).std() * np.sqrt(252)

        vol_q1 = rolling_vol.quantile(0.33)
        vol_q2 = rolling_vol.quantile(0.66)

        regime = pd.Series(index=self.returns.index, dtype="object")

        for i in regime.index:
            r = rolling_ret.loc[i]
            v = rolling_vol.loc[i]
            if pd.isna(r) or pd.isna(v):
                regime.loc[i] = None
            elif r > 0 and v <= vol_q1:
                regime.loc[i] = "Bull / Low Vol"
            elif r < 0 and v >= vol_q2:
                regime.loc[i] = "Bear / High Vol"
            else:
                regime.loc[i] = "Neutral"

        out = pd.DataFrame({
            "return_ann": rolling_ret,
            "vol_ann": rolling_vol,
            "regime": regime
        }).dropna()

        return out
