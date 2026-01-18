from pl_calibrator.data import load_matches_csv
from pl_calibrator.features import add_outcome_label, add_simple_form_features, build_feature_matrix
from pl_calibrator.model import fit_probability_model


def test_pipeline_smoke(tmp_path):
    # Write a tiny CSV to a temp file
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "date,home_team,away_team,home_goals,away_goals,market_prob_homewin\n"
        "2024-08-01,A,B,2,0,0.70\n"  # 1
        "2024-08-02,C,D,0,1,0.45\n"  # 0
        "2024-08-03,A,C,1,1,0.55\n"  # 0
        "2024-08-04,B,D,2,1,0.52\n"  # 1
        "2024-08-05,C,A,3,1,0.58\n"  # 1
        "2024-08-06,D,B,0,2,0.42\n"  # 0
        "2024-08-07,A,D,1,3,0.65\n"  # 0
        "2024-08-08,B,C,2,1,0.48\n"  # 1
        "2024-08-09,C,B,1,2,0.50\n"  # 0
        "2024-08-10,D,A,2,0,0.40\n"  # 1
        "2024-08-11,A,B,0,2,0.62\n"  # 0
        "2024-08-12,C,D,2,0,0.57\n"  # 1
        "2024-08-13,A,C,3,2,0.60\n"  # 1
        "2024-08-14,B,D,1,1,0.49\n"  # 0
        "2024-08-15,C,A,0,1,0.44\n"  # 0
        "2024-08-16,D,B,3,1,0.53\n"  # 1
        "2024-08-17,A,D,2,2,0.54\n"  # 0
        "2024-08-18,B,C,0,1,0.46\n"  # 0
        "2024-08-19,C,B,2,1,0.55\n"  # 1
        "2024-08-20,D,A,0,2,0.41\n"  # 0
    )

    df = load_matches_csv(str(csv_path))
    df = add_outcome_label(df)
    df = add_simple_form_features(df, n=3)
    df, feature_cols = build_feature_matrix(df)

    fit = fit_probability_model(df, feature_cols)
    assert 0.0 <= fit.brier_test <= 1.0
