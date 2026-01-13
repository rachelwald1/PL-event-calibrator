from __future__ import annotations

import numpy as np
import pandas as pd


def infer_regime(df: pd.DataFrame) -> pd.Series:
    """
    Simple regime classifier for MVP.

    Regime idea:
    - "stable" when market probabilities are not whipsawing
    - "volatile" when they are changing a lot

    This is a *proxy* for contextual uncertainty.
    You can later replace it with richer regimes (season stage, congestion, etc).
    """
    out = df.copy()

    # Rolling standard deviation of market probabilities by home team (simple proxy)
    vol = (
        out.groupby("home_team")["market_prob_homewin"]
           .transform(lambda s: s.shift(1).rolling(8, min_periods=3).std())
    ).fillna(0.0)

    regime = np.where(vol > 0.08, "volatile", "stable")
    return pd.Series(regime, index=df.index, name="regime")


def alpha_for_regime(regime: str) -> float:
    """
    How much to trust the *model* vs the *market* in each regime.

    alpha = 1.0 -> fully model
    alpha = 0.0 -> fully market

    MVP heuristic:
    - stable: trust model more
    - volatile: shrink model towards market
    """
    return 0.65 if regime == "stable" else 0.35
