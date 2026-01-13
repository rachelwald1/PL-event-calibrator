from __future__ import annotations

import pandas as pd


REQUIRED_COLS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "market_prob_homewin",
]


def load_matches_csv(path: str) -> pd.DataFrame:
    """
    Load match rows and validate required columns.

    This is deliberately strict: if the dataset is malformed, we fail early.
    """
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Parse & sort by date so rolling features work correctly
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Basic clean-up
    for c in ["home_goals", "away_goals"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # Market probabilities must be floats in (0,1)
    df["market_prob_homewin"] = pd.to_numeric(df["market_prob_homewin"], errors="raise")
    if (df["market_prob_homewin"] <= 0).any() or (df["market_prob_homewin"] >= 1).any():
        raise ValueError("market_prob_homewin must be strictly between 0 and 1")

    return df
