from __future__ import annotations

import numpy as np
import pandas as pd


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define the event: 'Home team wins?'
    y_homewin = 1 if home_goals > away_goals else 0
    """
    out = df.copy()
    out["y_homewin"] = (out["home_goals"] > out["away_goals"]).astype(int)
    return out


def add_simple_form_features(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Very simple strength proxies using rolling points form.

    - Compute points earned by the HOME team in that match (3/1/0)
    - Compute points earned by the AWAY team in that match (3/1/0)
    - Then for each team, compute rolling mean points over last n matches

    This is a lightweight substitute for Elo ratings while you build the MVP.
    """
    out = df.copy()

    # Points in this match (from each side's perspective)
    out["home_points"] = np.where(out["home_goals"] > out["away_goals"], 3,
                           np.where(out["home_goals"] == out["away_goals"], 1, 0))
    out["away_points"] = np.where(out["away_goals"] > out["home_goals"], 3,
                           np.where(out["away_goals"] == out["home_goals"], 1, 0))

    # Rolling form: average points in last n matches (shifted so we only use past info)
    out["home_form_pts"] = (
        out.groupby("home_team")["home_points"]
           .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    )
    out["away_form_pts"] = (
        out.groupby("away_team")["away_points"]
           .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
    )

    out["form_diff"] = out["home_form_pts"] - out["away_form_pts"]

    # Keep feature set explicit and small for MVP
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Finalise feature columns used by the probability model.

    MVP features:
    - market_prob_homewin (lets model learn how to interpret the market)
    - home_form_pts, away_form_pts, form_diff (simple team-strength proxy)

    Later you can add:
    - Elo ratings
    - rest days / congestion
    - season stage
    - injuries proxies
    """
    out = df.copy()

    feature_cols = [
        "market_prob_homewin",
        "home_form_pts",
        "away_form_pts",
        "form_diff",
    ]

    out = out.dropna(subset=feature_cols + ["y_homewin"])
    return out, feature_cols
