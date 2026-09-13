from pathlib import Path

import pandas as pd

import src.decision_engine.next_best_action as nba_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime"
)

GOVERNANCE_FILE = (
    RUNTIME_DIR
    / "cross_sell_model_governance.parquet"
)

MODEL_SCORE_FILE = (
    RUNTIME_DIR
    / "cross_sell_model_scores.parquet"
)

CATEGORY_RECOMMENDATION_FILE = (
    RUNTIME_DIR
    / "cross_sell_category_recommendations.parquet"
)

CATEGORY_VALIDATION_FILE = (
    RUNTIME_DIR
    / "cross_sell_category_validation.parquet"
)

NBA_FILE = (
    RUNTIME_DIR
    / "customer_next_best_action.parquet"
)


def test_cross_sell_model_passes_all_governance_gates():

    governance = pd.read_parquet(
        GOVERNANCE_FILE
    )

    assert not governance.empty
    assert "pass" in governance.columns

    passed = governance["pass"]

    if passed.dtype != bool:
        passed = (
            passed
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        )

    assert passed.all()


def test_cross_sell_propensity_scores_are_governed_and_bounded():

    scores = pd.read_parquet(
        MODEL_SCORE_FILE
    )

    assert not scores.empty

    assert scores[
        "golden_customer_id"
    ].is_unique

    assert set(
        scores[
            "model_status"
        ]
        .dropna()
        .unique()
    ) == {
        "ACCEPTED"
    }

    assert scores[
        "cross_sell_propensity_180d"
    ].between(
        0,
        1,
        inclusive="both",
    ).all()

    assert scores[
        "eligible_for_activation"
    ].all()


def test_cross_sell_category_recommendations_are_unique_and_scored():

    recommendations = pd.read_parquet(
        CATEGORY_RECOMMENDATION_FILE
    )

    scores = pd.read_parquet(
        MODEL_SCORE_FILE
    )

    assert not recommendations.empty

    assert recommendations[
        "golden_customer_id"
    ].is_unique

    assert recommendations[
        "recommended_category"
    ].notna().all()

    assert recommendations[
        "category_affinity_score"
    ].between(
        0,
        100,
        inclusive="both",
    ).all()

    assert set(
        recommendations[
            "golden_customer_id"
        ]
    ).issubset(
        set(
            scores[
                "golden_customer_id"
            ]
        )
    )


def test_cross_sell_category_validation_has_useful_top_n_accuracy():

    validation = pd.read_parquet(
        CATEGORY_VALIDATION_FILE
    )

    assert not validation.empty

    positive = validation.loc[
        validation[
            "actual_cross_sell"
        ].eq(1)
    ].copy()

    assert not positive.empty

    top1 = positive[
        "top1_hit"
    ].mean()

    top2 = positive[
        "top2_hit"
    ].mean()

    top3 = positive[
        "top3_hit"
    ].mean()

    assert 0 <= top1 <= top2 <= top3 <= 1

    # These are deliberately material but not over-fitted thresholds.
    assert top1 >= 0.50
    assert top2 >= 0.75
    assert top3 >= 0.90


def test_nba_uses_governed_cross_sell_model_without_breaking_fallback():

    nba = pd.read_parquet(
        NBA_FILE
    )

    assert not nba.empty

    assert nba[
        "golden_customer_id"
    ].is_unique

    assert (
        nba[
            "cross_sell_model_accepted"
        ]
        .fillna(False)
        .all()
    )

    model_rows = nba.loc[
        nba[
            "cross_sell_propensity_source"
        ]
        .eq(
            "Accepted governed model"
        )
    ].copy()

    assert not model_rows.empty

    assert model_rows[
        "cross_sell_propensity"
    ].between(
        0,
        1,
        inclusive="both",
    ).all()

    cross_sell = nba.loc[
        nba[
            "recommended_action_nba"
        ]
        .eq(
            "Cross-sell"
        )
    ].copy()

    assert not cross_sell.empty

    # Some cross-sell recommendations can legitimately retain the
    # transparent fallback if the governed model did not score them.
    assert set(
        cross_sell[
            "cross_sell_propensity_source"
        ]
        .dropna()
        .unique()
    ).issubset(
        {
            "Accepted governed model",
            "Rule-based fallback",
        }
    )

    interventions = nba.loc[
        ~nba[
            "recommended_action_nba"
        ]
        .eq(
            "Do Nothing"
        )
    ]

    assert (
        interventions[
            "expected_incremental_margin"
        ]
        .gt(0)
        .all()
    )


def test_nba_model_status_loader_falls_back_when_governance_missing(
    monkeypatch,
    tmp_path,
):

    missing_file = (
        tmp_path
        / "missing_governance.parquet"
    )

    monkeypatch.setattr(
        nba_module,
        "CROSS_SELL_GOVERNANCE_FILE",
        missing_file,
    )

    assert (
        nba_module
        .load_cross_sell_model_status()
        is False
    )
