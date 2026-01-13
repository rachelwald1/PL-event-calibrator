from __future__ import annotations

import argparse

import pandas as pd

from pl_calibrator.data import load_matches_csv
from pl_calibrator.features import add_outcome_label, add_simple_form_features, build_feature_matrix
from pl_calibrator.regime import infer_regime, alpha_for_regime
from pl_calibrator.model import fit_probability_model, predict_model_prob


def main() -> int:
    parser = argparse.ArgumentParser(description="Premier League event probability calibration (MVP).")
    parser.add_argument("--csv", required=True, help="Path to matches CSV.")
    parser.add_argument("--out", default="outputs_calibrated.csv", help="Output CSV path.")
    args = parser.parse_args()

    # 1) Load
    df = load_matches_csv(args.csv)

    # 2) Create label + features
    df = add_outcome_label(df)
    df = add_simple_form_features(df, n=5)
    df, feature_cols = build_feature_matrix(df)

    # 3) Fit probability model
    fit = fit_probability_model(df, feature_cols)

    print("\n=== Calibration metrics (prototype) ===")
    print(f"Brier score (lower better): {fit.brier_test:.4f}")
    print("\nReliability table (p_mean vs y_rate):")
    if len(fit.reliability) == 0:
        print("Not enough data for bins (need more rows).")
    else:
        print(fit.reliability.to_string(index=False))

    # 4) Produce outputs per match
    df["p_market"] = df["market_prob_homewin"]
    df["p_model"] = predict_model_prob(fit, df)

    df["regime"] = infer_regime(df)
    df["alpha"] = df["regime"].map(alpha_for_regime).astype(float)

    # Regime-adjusted probability = blend(model, market)
    df["p_adjusted"] = df["alpha"] * df["p_model"] + (1 - df["alpha"]) * df["p_market"]
    df["prob_gap"] = df["p_adjusted"] - df["p_market"]

    # 5) Save
    cols = [
        "date", "home_team", "away_team",
        "p_market", "p_model", "regime", "alpha", "p_adjusted", "prob_gap",
        "y_homewin"
    ]
    df_out = df[cols].copy()
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("\nSample rows:")
    print(df_out.tail(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
