from pathlib import Path

import pandas as pd


NBA_FILE = Path(
    "data/runtime/customer_next_best_action.parquet"
)

VALID_ACTIONS = {
    "Protect",
    "Re-engage",
    "Develop",
    "Cross-sell",
    "Promote",
    "Do Nothing",
}


def test_nba_has_one_recommendation_per_customer():
    nba = pd.read_parquet(NBA_FILE)

    assert len(nba) == 20_000
    assert nba["customer_id"].nunique() == 20_000
    assert not nba["customer_id"].duplicated().any()


def test_nba_action_taxonomy_and_bounds_are_valid():
    nba = pd.read_parquet(NBA_FILE)

    assert set(
        nba["recommended_action_nba"].dropna().unique()
    ).issubset(VALID_ACTIONS)

    assert nba[
        "recommendation_confidence"
    ].between(0, 100).all()

    assert nba[
        "response_probability"
    ].between(0, 1).all()


def test_recommended_interventions_have_positive_economics():
    nba = pd.read_parquet(NBA_FILE)

    interventions = nba.loc[
        ~nba["recommended_action_nba"].eq(
            "Do Nothing"
        )
    ]

    assert not interventions.empty

    assert (
        interventions["expected_incremental_margin"]
        > 0
    ).all()

    assert (
        interventions["expected_incremental_sales"]
        >= 0
    ).all()


def test_do_nothing_has_zero_incremental_economics():
    nba = pd.read_parquet(NBA_FILE)

    do_nothing = nba.loc[
        nba["recommended_action_nba"].eq(
            "Do Nothing"
        )
    ]

    assert not do_nothing.empty

    assert (
        do_nothing["expected_incremental_margin"]
        == 0
    ).all()

    assert (
        do_nothing["expected_incremental_sales"]
        == 0
    ).all()


def test_recommendation_explainability_is_populated():
    nba = pd.read_parquet(NBA_FILE)

    required_text_columns = [
        "primary_driver",
        "alternative_action",
        "recommendation_rationale",
    ]

    for column in required_text_columns:
        assert nba[column].notna().all()
        assert nba[column].astype(str).str.strip().ne("").all()

    assert set(
        nba["alternative_action"].dropna().unique()
    ).issubset(VALID_ACTIONS)
