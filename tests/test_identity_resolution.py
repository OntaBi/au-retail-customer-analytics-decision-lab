from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

IDENTITY_SOURCE_PATH = (
    ROOT
    / "data"
    / "generated"
    / "customer_identity_records.parquet"
)

GROUND_TRUTH_PATH = (
    ROOT
    / "data"
    / "generated"
    / "identity_ground_truth.parquet"
)

RESOLUTION_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "customer_identity_resolution.parquet"
)

SUMMARY_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "identity_resolution_summary.parquet"
)

EDGE_AUDIT_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "identity_edge_audit.parquet"
)


def load_summary():
    summary = pd.read_parquet(
        SUMMARY_PATH
    )

    return dict(
        zip(
            summary["metric"],
            summary["value"],
        )
    )


def test_identity_resolution_has_one_output_per_source_record():
    source = pd.read_parquet(
        IDENTITY_SOURCE_PATH
    )

    resolution = pd.read_parquet(
        RESOLUTION_PATH
    )

    assert len(resolution) == len(source)

    assert (
        resolution["identity_record_id"]
        .is_unique
    )

    assert (
        resolution["golden_customer_id"]
        .notna()
        .all()
    )


def test_identity_resolution_quality_thresholds():
    metrics = load_summary()

    assert float(
        metrics["precision"]
    ) >= 0.985

    assert float(
        metrics["recall"]
    ) >= 0.95

    assert float(
        metrics["false_merge_rate"]
    ) <= 0.015

    assert float(
        metrics["missed_link_rate"]
    ) <= 0.05


def test_identity_resolution_does_not_collapse_population():
    truth = pd.read_parquet(
        GROUND_TRUTH_PATH
    )

    resolution = pd.read_parquet(
        RESOLUTION_PATH
    )

    true_customers = (
        truth["golden_customer_id"]
        .nunique()
    )

    predicted_customers = (
        resolution["golden_customer_id"]
        .nunique()
    )

    # Conservative resolution may create slightly
    # more golden identities than ground truth,
    # but should remain close to the expected population.
    difference = abs(
        predicted_customers
        - true_customers
    )

    assert difference / true_customers <= 0.03


def test_edge_audit_contains_accepted_and_rejected_decisions():
    audit = pd.read_parquet(
        EDGE_AUDIT_PATH
    )

    assert not audit.empty

    assert {
        "record_a",
        "record_b",
        "match_method",
        "score",
        "accepted",
        "rejection_reason",
    }.issubset(
        audit.columns
    )

    assert (
        audit["accepted"].any()
    )

    assert (
        (~audit["accepted"]).any()
    )


def test_ground_truth_is_not_present_in_runtime_resolution():
    resolution = pd.read_parquet(
        RESOLUTION_PATH
    )

    prohibited = {
        "true_person_id",
        "true_golden_customer_id",
        "variant_type",
        "household_case_id",
    }

    assert prohibited.isdisjoint(
        set(resolution.columns)
    )


def test_shared_households_are_not_collapsed():
    source = pd.read_parquet(
        IDENTITY_SOURCE_PATH
    )

    truth = pd.read_parquet(
        GROUND_TRUTH_PATH
    ).rename(
        columns={
            "golden_customer_id":
            "true_golden_customer_id"
        }
    )

    resolution = pd.read_parquet(
        RESOLUTION_PATH
    )

    households = (
        source[
            source["household_case_id"]
            .notna()
        ][
            [
                "identity_record_id",
                "household_case_id",
            ]
        ]
        .merge(
            truth,
            on="identity_record_id",
            how="left",
        )
        .merge(
            resolution[
                [
                    "identity_record_id",
                    "golden_customer_id",
                ]
            ],
            on="identity_record_id",
            how="left",
        )
    )

    case_summary = (
        households
        .groupby(
            "household_case_id"
        )
        .agg(
            true_people=(
                "true_golden_customer_id",
                "nunique",
            ),
            predicted_people=(
                "golden_customer_id",
                "nunique",
            ),
        )
    )

    incorrect_merges = (
        case_summary[
            "predicted_people"
        ]
        < case_summary[
            "true_people"
        ]
    ).sum()

    assert incorrect_merges == 0