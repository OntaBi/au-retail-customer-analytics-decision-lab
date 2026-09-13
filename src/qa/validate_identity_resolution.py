from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IDENTITY_SOURCE_PATH = (
    ROOT / "data" / "generated" / "customer_identity_records.parquet"
)

GROUND_TRUTH_PATH = (
    ROOT / "data" / "generated" / "identity_ground_truth.parquet"
)

RESOLUTION_PATH = (
    ROOT / "data" / "runtime" / "customer_identity_resolution.parquet"
)

DIAGNOSTIC_OUTPUT_PATH = (
    ROOT / "data" / "runtime" / "identity_false_merge_diagnostics.parquet"
)

CONTAMINATED_CLUSTER_PATH = (
    ROOT / "data" / "runtime" / "identity_contaminated_clusters.parquet"
)


# ============================================================
# Load evaluation frame
# ============================================================

def load_evaluation_data() -> pd.DataFrame:
    for path in [
        IDENTITY_SOURCE_PATH,
        GROUND_TRUTH_PATH,
        RESOLUTION_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    source = pd.read_parquet(
        IDENTITY_SOURCE_PATH
    )

    ground_truth = pd.read_parquet(
        GROUND_TRUTH_PATH
    ).rename(
        columns={
            "golden_customer_id": "true_person_id_qa"
        }
    )

    resolution = pd.read_parquet(
        RESOLUTION_PATH
    )

    evaluation = (
        resolution
        .merge(
            ground_truth[
                [
                    "identity_record_id",
                    "true_person_id_qa",
                ]
            ],
            on="identity_record_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            source[
                [
                    c
                    for c in [
                        "identity_record_id",
                        "variant_type",
                        "household_case_id",
                    ]
                    if c in source.columns
                ]
            ],
            on="identity_record_id",
            how="left",
            validate="one_to_one",
        )
    )

    if evaluation["true_person_id_qa"].isna().any():
        missing = (
            evaluation["true_person_id_qa"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{missing:,} resolved records have no "
            f"ground-truth identity."
        )

    return evaluation


# ============================================================
# Cluster analysis
# ============================================================

def build_cluster_summary(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    cluster_summary = (
        evaluation
        .groupby("golden_customer_id")
        .agg(
            records_in_cluster=(
                "identity_record_id",
                "size",
            ),
            true_people_in_cluster=(
                "true_person_id_qa",
                "nunique",
            ),
            mean_match_confidence=(
                "match_confidence",
                "mean",
            ),
            min_match_confidence=(
                "match_confidence",
                "min",
            ),
            ambiguous_records=(
                "ambiguous_match_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    cluster_summary["contaminated_cluster"] = (
        cluster_summary["true_people_in_cluster"] > 1
    )

    return cluster_summary


def add_cluster_diagnostics(
    evaluation: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    result = evaluation.merge(
        cluster_summary,
        on="golden_customer_id",
        how="left",
        validate="many_to_one",
    )

    return result


# ============================================================
# Method diagnostics
# ============================================================

def build_method_diagnostics(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics = (
        evaluation
        .groupby("match_method")
        .agg(
            records=(
                "identity_record_id",
                "size",
            ),
            contaminated_records=(
                "contaminated_cluster",
                "sum",
            ),
            mean_confidence=(
                "match_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    diagnostics["clean_records"] = (
        diagnostics["records"]
        - diagnostics["contaminated_records"]
    )

    diagnostics["contamination_rate"] = np.where(
        diagnostics["records"] > 0,
        (
            diagnostics["contaminated_records"]
            / diagnostics["records"]
        ),
        0.0,
    )

    diagnostics = diagnostics.sort_values(
        [
            "contamination_rate",
            "contaminated_records",
        ],
        ascending=[False, False],
    )

    return diagnostics


# ============================================================
# Household diagnostics
# ============================================================

def build_household_diagnostics(
    evaluation: pd.DataFrame,
) -> dict:
    if "household_case_id" not in evaluation.columns:
        return {
            "household_records": 0,
            "household_cases": 0,
            "household_cases_merged": 0,
            "household_merge_rate": 0.0,
        }

    household = evaluation[
        evaluation["household_case_id"].notna()
    ].copy()

    if household.empty:
        return {
            "household_records": 0,
            "household_cases": 0,
            "household_cases_merged": 0,
            "household_merge_rate": 0.0,
        }

    case_summary = (
        household
        .groupby("household_case_id")
        .agg(
            true_people=(
                "true_person_id_qa",
                "nunique",
            ),
            predicted_clusters=(
                "golden_customer_id",
                "nunique",
            ),
            records=(
                "identity_record_id",
                "size",
            ),
        )
        .reset_index()
    )

    # A synthetic household normally contains two distinct
    # people. If they collapse into fewer predicted clusters
    # than true people, the resolver has merged household
    # identities.
    case_summary["incorrect_household_merge"] = (
        case_summary["predicted_clusters"]
        < case_summary["true_people"]
    )

    household_cases = len(case_summary)

    household_cases_merged = int(
        case_summary["incorrect_household_merge"]
        .sum()
    )

    return {
        "household_records": len(household),
        "household_cases": household_cases,
        "household_cases_merged": household_cases_merged,
        "household_merge_rate": (
            household_cases_merged
            / household_cases
            if household_cases
            else 0.0
        ),
    }


# ============================================================
# False-pair analysis
# ============================================================

def count_false_pairs_by_cluster(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for golden_id, group in evaluation.groupby(
        "golden_customer_id"
    ):
        if group["true_person_id_qa"].nunique() <= 1:
            continue

        true_counts = (
            group["true_person_id_qa"]
            .value_counts()
        )

        n = len(group)

        total_predicted_pairs = (
            n * (n - 1)
        ) // 2

        true_positive_pairs = sum(
            (count * (count - 1)) // 2
            for count in true_counts
        )

        false_pairs = (
            total_predicted_pairs
            - true_positive_pairs
        )

        rows.append(
            {
                "golden_customer_id": golden_id,
                "records_in_cluster": n,
                "true_people_in_cluster": (
                    len(true_counts)
                ),
                "false_pairs_generated": (
                    false_pairs
                ),
                "largest_true_identity_records": (
                    int(true_counts.max())
                ),
                "mean_match_confidence": (
                    group["match_confidence"]
                    .mean()
                ),
                "min_match_confidence": (
                    group["match_confidence"]
                    .min()
                ),
                "match_methods": " | ".join(
                    sorted(
                        group["match_method"]
                        .dropna()
                        .unique()
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "false_pairs_generated",
            ascending=False,
        )

    return result


# ============================================================
# Variant diagnostics
# ============================================================

def build_variant_diagnostics(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    if "variant_type" not in evaluation.columns:
        return pd.DataFrame()

    diagnostics = (
        evaluation
        .groupby("variant_type")
        .agg(
            records=(
                "identity_record_id",
                "size",
            ),
            contaminated_records=(
                "contaminated_cluster",
                "sum",
            ),
        )
        .reset_index()
    )

    diagnostics["contamination_rate"] = np.where(
        diagnostics["records"] > 0,
        (
            diagnostics["contaminated_records"]
            / diagnostics["records"]
        ),
        0.0,
    )

    return diagnostics.sort_values(
        "contamination_rate",
        ascending=False,
    )


# ============================================================
# Diagnostic output
# ============================================================

def build_diagnostic_output(
    evaluation: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    household_metrics: dict,
) -> pd.DataFrame:
    contaminated_clusters = cluster_summary[
        cluster_summary["contaminated_cluster"]
    ]

    contaminated_records = evaluation[
        evaluation["contaminated_cluster"]
    ]

    max_true_people = (
        int(
            contaminated_clusters[
                "true_people_in_cluster"
            ].max()
        )
        if not contaminated_clusters.empty
        else 0
    )

    largest_cluster = (
        int(
            contaminated_clusters[
                "records_in_cluster"
            ].max()
        )
        if not contaminated_clusters.empty
        else 0
    )

    values = {
        "source_records": len(evaluation),
        "predicted_golden_customers": (
            evaluation["golden_customer_id"]
            .nunique()
        ),
        "contaminated_clusters": (
            len(contaminated_clusters)
        ),
        "records_in_contaminated_clusters": (
            len(contaminated_records)
        ),
        "largest_contaminated_cluster": (
            largest_cluster
        ),
        "max_true_people_in_one_cluster": (
            max_true_people
        ),
        **household_metrics,
    }

    return pd.DataFrame(
        {
            "metric": list(values.keys()),
            "value": list(values.values()),
        }
    )


# ============================================================
# Printing
# ============================================================

def print_report(
    evaluation: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    method_diagnostics: pd.DataFrame,
    household_metrics: dict,
    false_pair_clusters: pd.DataFrame,
    variant_diagnostics: pd.DataFrame,
):
    contaminated = cluster_summary[
        cluster_summary["contaminated_cluster"]
    ]

    print()
    print("=" * 78)
    print("FALSE MERGE DIAGNOSTICS")
    print("=" * 78)

    print(
        f"Predicted golden customers         : "
        f"{evaluation['golden_customer_id'].nunique():,}"
    )

    print(
        f"Contaminated clusters              : "
        f"{len(contaminated):,}"
    )

    contaminated_records = evaluation[
        evaluation["contaminated_cluster"]
    ]

    print(
        f"Records in contaminated clusters  : "
        f"{len(contaminated_records):,}"
    )

    largest_cluster = (
        contaminated["records_in_cluster"].max()
        if not contaminated.empty
        else 0
    )

    print(
        f"Largest contaminated cluster       : "
        f"{int(largest_cluster):,}"
    )

    max_true_people = (
        contaminated["true_people_in_cluster"].max()
        if not contaminated.empty
        else 0
    )

    print(
        f"Max true people in one cluster     : "
        f"{int(max_true_people):,}"
    )

    print()
    print("MATCH METHOD CONTAMINATION")
    print("-" * 78)

    method_print = method_diagnostics.copy()

    method_print["contamination_rate"] = (
        method_print["contamination_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    method_print["mean_confidence"] = (
        method_print["mean_confidence"]
        .map(lambda x: f"{x:.2%}")
    )

    print(
        method_print[
            [
                "match_method",
                "records",
                "contaminated_records",
                "contamination_rate",
                "mean_confidence",
            ]
        ].to_string(index=False)
    )

    print()
    print("SHARED-HOUSEHOLD PERFORMANCE")
    print("-" * 78)

    print(
        f"Household records                  : "
        f"{household_metrics['household_records']:,}"
    )

    print(
        f"Household cases                    : "
        f"{household_metrics['household_cases']:,}"
    )

    print(
        f"Household cases incorrectly merged : "
        f"{household_metrics['household_cases_merged']:,}"
    )

    print(
        f"Household false-merge rate         : "
        f"{household_metrics['household_merge_rate']:.2%}"
    )

    print()
    print("CLUSTER AMPLIFICATION")
    print("-" * 78)

    if false_pair_clusters.empty:
        print(
            "No contaminated clusters detected."
        )

    else:
        total_false_pairs = (
            false_pair_clusters[
                "false_pairs_generated"
            ].sum()
        )

        top_10_false_pairs = (
            false_pair_clusters
            .head(10)["false_pairs_generated"]
            .sum()
        )

        print(
            f"False pairs from contaminated "
            f"clusters : {total_false_pairs:,}"
        )

        print(
            f"False pairs from top 10 clusters  : "
            f"{top_10_false_pairs:,}"
        )

        print()
        print("TOP CONTAMINATED CLUSTERS")
        print("-" * 78)

        top = false_pair_clusters.head(15).copy()

        top["mean_match_confidence"] = (
            top["mean_match_confidence"]
            .map(lambda x: f"{x:.2%}")
        )

        top["min_match_confidence"] = (
            top["min_match_confidence"]
            .map(lambda x: f"{x:.2%}")
        )

        print(
            top[
                [
                    "golden_customer_id",
                    "records_in_cluster",
                    "true_people_in_cluster",
                    "false_pairs_generated",
                    "mean_match_confidence",
                    "min_match_confidence",
                    "match_methods",
                ]
            ].to_string(index=False)
        )

    if not variant_diagnostics.empty:
        print()
        print("SYNTHETIC VARIANT CONTAMINATION")
        print("-" * 78)

        variant_print = (
            variant_diagnostics
            .head(15)
            .copy()
        )

        variant_print[
            "contamination_rate"
        ] = (
            variant_print[
                "contamination_rate"
            ]
            .map(lambda x: f"{x:.2%}")
        )

        print(
            variant_print.to_string(
                index=False
            )
        )

    print()
    print(
        f"Diagnostics saved to: "
        f"{DIAGNOSTIC_OUTPUT_PATH}"
    )

    print(
        f"Contaminated clusters: "
        f"{CONTAMINATED_CLUSTER_PATH}"
    )

    print("=" * 78)


# ============================================================
# Main
# ============================================================

def main():
    evaluation = load_evaluation_data()

    cluster_summary = build_cluster_summary(
        evaluation
    )

    evaluation = add_cluster_diagnostics(
        evaluation,
        cluster_summary,
    )

    method_diagnostics = (
        build_method_diagnostics(
            evaluation
        )
    )

    household_metrics = (
        build_household_diagnostics(
            evaluation
        )
    )

    false_pair_clusters = (
        count_false_pairs_by_cluster(
            evaluation
        )
    )

    variant_diagnostics = (
        build_variant_diagnostics(
            evaluation
        )
    )

    diagnostic_output = (
        build_diagnostic_output(
            evaluation,
            cluster_summary,
            household_metrics,
        )
    )

    DIAGNOSTIC_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_output.to_parquet(
        DIAGNOSTIC_OUTPUT_PATH,
        index=False,
    )

    false_pair_clusters.to_parquet(
        CONTAMINATED_CLUSTER_PATH,
        index=False,
    )

    print_report(
        evaluation,
        cluster_summary,
        method_diagnostics,
        household_metrics,
        false_pair_clusters,
        variant_diagnostics,
    )


if __name__ == "__main__":
    main()