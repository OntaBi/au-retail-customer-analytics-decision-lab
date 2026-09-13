from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CUSTOMER_MASTER_FILE = Path(
    "data/generated/customer_master.parquet"
)

IDENTITY_GROUND_TRUTH_FILE = Path(
    "data/generated/identity_ground_truth.parquet"
)

IDENTITY_RESOLUTION_FILE = Path(
    "data/runtime/customer_identity_resolution.parquet"
)

CUSTOMER_FEATURE_FILE = Path(
    "data/runtime/customer_features.parquet"
)

OUTPUT_DIR = Path("outputs")

CROSSTAB_FILE = OUTPUT_DIR / "lapse_model_crosstab.csv"
PROFILE_FILE = OUTPUT_DIR / "lapse_model_profile.csv"


# ---------------------------------------------------------------------
# Build validation dataset
# ---------------------------------------------------------------------


def build_golden_truth_bridge(
    customer_master: pd.DataFrame,
    identity_ground_truth: pd.DataFrame,
    identity_resolution: pd.DataFrame,
) -> pd.DataFrame:
    """
    QA-only bridge from resolved golden identities back to hidden
    synthetic truth. Production analytics never consume this bridge.

    Only uncontaminated resolved clusters are used for behavioural
    truth validation. This prevents a known false merge from being
    assigned an arbitrary synthetic archetype/persona.
    """
    record_truth = (
        identity_resolution[
            ["identity_record_id", "golden_customer_id"]
        ]
        .merge(
            identity_ground_truth[
                ["identity_record_id", "golden_customer_id"]
            ].rename(
                columns={
                    "golden_customer_id": "customer_id"
                }
            ),
            on="identity_record_id",
            how="left",
            validate="one_to_one",
        )
    )

    cluster_truth = (
        record_truth
        .groupby("golden_customer_id")
        .agg(
            true_people=("customer_id", "nunique"),
            customer_id=("customer_id", "first"),
        )
        .reset_index()
    )

    clean = cluster_truth.loc[
        cluster_truth["true_people"].eq(1)
    ].copy()

    truth = customer_master[
        [
            "customer_id",
            "archetype"
        ]
    ].copy()

    return (
        clean
        .merge(
            truth,
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
        .drop(columns=["customer_id", "true_people"])
    )

def build_validation_data(
    customer_master: pd.DataFrame,
    identity_ground_truth: pd.DataFrame,
    identity_resolution: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:

    truth = build_golden_truth_bridge(
        customer_master,
        identity_ground_truth,
        identity_resolution,
    )

    return truth.merge(
        customer_features,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )



# ---------------------------------------------------------------------
# Cross-tab validation
# ---------------------------------------------------------------------

def build_lifecycle_crosstab(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["archetype"],
        validation["lifecycle_status"],
        normalize="index",
    )

    crosstab = (
        crosstab
        .mul(100)
        .round(1)
    )

    return crosstab


# ---------------------------------------------------------------------
# Behavioural profile
# ---------------------------------------------------------------------

def build_archetype_profile(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    profile = (
        validation
        .groupby("archetype")
        .agg(
            customers=("golden_customer_id", "nunique"),
            active_rate=(
                "active_customer",
                "mean",
            ),
            median_days_since_purchase=(
                "days_since_last_purchase",
                "median",
            ),
            median_purchase_gap=(
                "median_purchase_gap_days",
                "median",
            ),
            median_lapse_ratio=(
                "adjusted_lapse_ratio",
                "median",
            ),
            median_cadence_cv=(
                "cadence_cv",
                "median",
            ),
            median_trailing_12m_sales=(
                "trailing_12m_sales",
                "median",
            ),
        )
        .reset_index()
    )

    profile["active_rate"] = (
        profile["active_rate"]
        .mul(100)
        .round(1)
    )

    numeric_columns = [
        "median_days_since_purchase",
        "median_purchase_gap",
        "median_lapse_ratio",
        "median_cadence_cv",
        "median_trailing_12m_sales",
    ]

    profile[numeric_columns] = (
        profile[numeric_columns]
        .round(2)
    )

    return profile


# ---------------------------------------------------------------------
# Key detection metrics
# ---------------------------------------------------------------------

def build_detection_summary(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    result = validation.copy()

    result["risk_detected"] = result[
        "lifecycle_status"
    ].isin(
        [
            "At Risk",
            "Highly Lapsed",
        ]
    )

    result["early_warning_detected"] = result[
        "lifecycle_status"
    ].isin(
        [
            "Watch",
            "At Risk",
            "Highly Lapsed",
        ]
    )

    summary = (
        result
        .groupby("archetype")
        .agg(
            customers=("golden_customer_id", "nunique"),
            risk_detection_rate=(
                "risk_detected",
                "mean",
            ),
            early_warning_rate=(
                "early_warning_detected",
                "mean",
            ),
        )
        .reset_index()
    )

    summary[
        [
            "risk_detection_rate",
            "early_warning_rate",
        ]
    ] = (
        summary[
            [
                "risk_detection_rate",
                "early_warning_rate",
            ]
        ]
        .mul(100)
        .round(1)
    )

    return summary


# ---------------------------------------------------------------------
# QA output
# ---------------------------------------------------------------------

def print_validation_results(
    validation: pd.DataFrame,
    crosstab: pd.DataFrame,
    profile: pd.DataFrame,
    detection_summary: pd.DataFrame,
) -> None:

    print("\nLAPSE MODEL VALIDATION")
    print("=" * 80)

    print(
        f"Customers validated: "
        f"{validation['golden_customer_id'].nunique():,}"
    )

    print("\nLIFECYCLE STATUS BY TRUE ARCHETYPE (%)")
    print("-" * 80)

    print(
        crosstab.to_string()
    )

    print("\nARCHETYPE PROFILE")
    print("-" * 80)

    print(
        profile.to_string(index=False)
    )

    print("\nRISK DETECTION SUMMARY (%)")
    print("-" * 80)

    print(
        detection_summary.to_string(index=False)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading validation data...")

    customer_master = pd.read_parquet(
        CUSTOMER_MASTER_FILE
    )

    identity_ground_truth = pd.read_parquet(
        IDENTITY_GROUND_TRUTH_FILE
    )

    identity_resolution = pd.read_parquet(
        IDENTITY_RESOLUTION_FILE
    )

    customer_features = pd.read_parquet(
        CUSTOMER_FEATURE_FILE
    )

    validation = build_validation_data(
        customer_master,
        identity_ground_truth,
        identity_resolution,
        customer_features,
    )

    crosstab = build_lifecycle_crosstab(
        validation
    )

    profile = build_archetype_profile(
        validation
    )

    detection_summary = build_detection_summary(
        validation
    )

    crosstab.to_csv(
        CROSSTAB_FILE
    )

    profile.to_csv(
        PROFILE_FILE,
        index=False,
    )

    print_validation_results(
        validation,
        crosstab,
        profile,
        detection_summary,
    )

    print("\nFiles created:")
    print(CROSSTAB_FILE)
    print(PROFILE_FILE)


if __name__ == "__main__":
    main()