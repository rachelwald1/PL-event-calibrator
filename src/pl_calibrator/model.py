from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


@dataclass
class FitResult:
    model: LogisticRegression
    feature_cols: list[str]
    brier_test: float
    reliability: pd.DataFrame


def reliability_table(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Calibration diagnostic:
    - bin predicted probabilities
    - compare mean predicted probability vs observed frequency in each bin
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1)
    b = np.digitize(p, edges) - 1
    b = np.clip(b, 0, bins - 1)

    rows = []
    for i in range(bins):
        m = b == i
        if m.sum() == 0:
            continue
        rows.append({
            "bin": i,
            "count": int(m.sum()),
            "p_mean": float(p[m].mean()),
            "y_rate": float(y_true[m].mean()),
        })
    return pd.DataFrame(rows)


def fit_probability_model(df: pd.DataFrame, feature_cols: list[str]) -> FitResult:
    """
    Fit logistic regression to estimate P(home_win).

    Why logistic regression?
    - interpretable learned weights (like your spam classifier)
    - outputs probabilities
    - strong baseline for small feature sets
    """
    X = df[feature_cols].values
    y = df["y_homewin"].values

    # MVP split: random stratified
    # Later: do a time-based split to avoid leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, p_test)
    rel = reliability_table(y_test, p_test, bins=10)

    return FitResult(model=model, feature_cols=feature_cols, brier_test=float(brier), reliability=rel)


def predict_model_prob(fit: FitResult, df: pd.DataFrame) -> np.ndarray:
    """Return p_model for each row."""
    X = df[fit.feature_cols].values
    p = fit.model.predict_proba(X)[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)
