from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PRIORITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "runtime"
    / "customer_priority.parquet"
)

CLUSTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "runtime"
    / "customer_clusters.parquet"
)


GOLDEN_MASTER_FILE = (
    PROJECT_ROOT / "data" / "runtime" / "golden_customer_master.parquet"
)

GOLDEN_TRANSACTION_FILE = (
    PROJECT_ROOT / "data" / "runtime" / "golden_customer_transactions.parquet"
)


VALID_CLUSTER_NAMES = {
    "Big Ticket Shoppers",
    "High Frequency Generalists",
    "Promotion-Led Shoppers",
    "Category Specialists",
    "Omnichannel Mainstream",
    "Store-Led Shoppers",
}

VALID_DECISION_GROUPS = {
    "Protect Now",
    "Proactive Retention",
    "Re-engage",
    "Watch Closely",
    "Develop",
    "Maintain",
    "Acquisition Opportunity",
    "Low Priority",
}


def test_customer_priority_has_unique_customer_records():

    priority = pd.read_parquet(
        PRIORITY_FILE
    )

    golden_master = pd.read_parquet(GOLDEN_MASTER_FILE)

    assert len(priority) == len(golden_master)
    assert priority["golden_customer_id"].is_unique
    assert set(priority["golden_customer_id"]) == set(
        golden_master["golden_customer_id"]
    )


def test_cluster_outputs_use_valid_final_segments():

    clusters = pd.read_parquet(
        CLUSTER_FILE
    )

    clustering_features = pd.read_parquet(
        PROJECT_ROOT
        / "data"
        / "runtime"
        / "clustering_features.parquet"
    )

    assert not clusters.empty

    assert clusters[
        "golden_customer_id"
    ].is_unique

    assert clustering_features[
        "golden_customer_id"
    ].is_unique

    assert set(
        clusters["golden_customer_id"]
    ) == set(
        clustering_features[
            "golden_customer_id"
        ]
    )

    assert set(
        clusters["cluster"]
        .dropna()
        .unique()
    ).issubset(
        set(range(6))
    )

    assert set(
        clusters["cluster_name"]
        .dropna()
        .unique()
    ) == VALID_CLUSTER_NAMES


def test_decision_scores_and_groups_are_valid():

    priority = pd.read_parquet(
        PRIORITY_FILE
    )

    score_columns = [
        "value_score",
        "lapse_risk_score",
        "momentum_risk_score",
        "priority_score",
    ]

    for column in score_columns:

        scores = (
            priority[column]
            .dropna()
        )

        assert (
            scores.between(
                0,
                100,
                inclusive="both",
            )
            .all()
        )

    assert set(
        priority[
            "decision_group"
        ]
        .dropna()
        .unique()
    ).issubset(
        VALID_DECISION_GROUPS
    )