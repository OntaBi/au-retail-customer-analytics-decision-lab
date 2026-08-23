from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CUSTOMER_MASTER_FILE = Path(
    "data/generated/customer_master.parquet"
)

MOMENTUM_FILE = Path(
    "data/runtime/customer_momentum.parquet"
)

OUTPUT_DIR = Path("outputs")

CROSSTAB_FILE = OUTPUT_DIR / "momentum_model_crosstab.csv"
PROFILE_FILE = OUTPUT_DIR / "momentum_model_profile.csv"


# ---------------------------------------------------------------------
# Build validation dataset
# ---------------------------------------------------------------------

def build_validation_data(
    customer_master: pd.DataFrame,
    momentum: pd.DataFrame,
) -> pd.DataFrame:

    truth = customer_master[
        [
            "customer_id",
            "archetype",
        ]
    ].copy()

    validation = truth.merge(
        momentum,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    return validation


# ---------------------------------------------------------------------
# Cross-tab
# ---------------------------------------------------------------------

def build_momentum_crosstab(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["archetype"],
        validation["momentum_status"],
        normalize="index",
    )

    return (
        crosstab
        .mul(100)
        .round(1)
    )


# ---------------------------------------------------------------------
# Archetype behavioural profile
# ---------------------------------------------------------------------

def build_archetype_profile(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    comparable = validation.loc[
        validation["comparable_history"]
    ].copy()

    profile = (
        comparable
        .groupby("archetype")
        .agg(
            comparable_customers=(
                "customer_id",
                "nunique",
            ),
            median_order_change=(
                "order_change_pct",
                "median",
            ),
            median_sales_change=(
                "sales_change_pct",
                "median",
            ),
            median_margin_change=(
                "margin_change_pct",
                "median",
            ),
            median_aov_change=(
                "aov_change_pct",
                "median",
            ),
            median_cadence_change=(
                "cadence_change_pct",
                "median",
            ),
        )
        .reset_index()
    )

    change_columns = [
        "median_order_change",
        "median_sales_change",
        "median_margin_change",
        "median_aov_change",
        "median_cadence_change",
    ]

    profile[change_columns] = (
        profile[change_columns]
        .mul(100)
        .round(1)
    )

    return profile


# ---------------------------------------------------------------------
# Detection summary
# ---------------------------------------------------------------------

def build_detection_summary(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    result = validation.copy()

    result["deterioration_detected"] = (
        result["momentum_status"]
        .isin(
            [
                "Deteriorating",
                "Strongly Deteriorating",
            ]
        )
    )

    result["strong_deterioration_detected"] = (
        result["momentum_status"]
        .eq("Strongly Deteriorating")
    )

    # Detection among all customers in the archetype.
    overall = (
        result
        .groupby("archetype")
        .agg(
            customers=("customer_id", "nunique"),
            deterioration_rate=(
                "deterioration_detected",
                "mean",
            ),
            strong_deterioration_rate=(
                "strong_deterioration_detected",
                "mean",
            ),
        )
        .reset_index()
    )

    # Detection among customers for whom period comparison is valid.
    comparable = result.loc[
        result["comparable_history"]
    ].copy()

    comparable_summary = (
        comparable
        .groupby("archetype")
        .agg(
            comparable_customers=(
                "customer_id",
                "nunique",
            ),
            comparable_deterioration_rate=(
                "deterioration_detected",
                "mean",
            ),
            comparable_strong_rate=(
                "strong_deterioration_detected",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = overall.merge(
        comparable_summary,
        on="archetype",
        how="left",
    )

    rate_columns = [
        "deterioration_rate",
        "strong_deterioration_rate",
        "comparable_deterioration_rate",
        "comparable_strong_rate",
    ]

    summary[rate_columns] = (
        summary[rate_columns]
        .mul(100)
        .round(1)
    )

    return summary


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

def print_validation_results(
    validation: pd.DataFrame,
    crosstab: pd.DataFrame,
    profile: pd.DataFrame,
    detection_summary: pd.DataFrame,
) -> None:

    print("\nMOMENTUM MODEL VALIDATION")
    print("=" * 90)

    print(
        f"Customers validated: "
        f"{validation['customer_id'].nunique():,}"
    )

    print("\nMOMENTUM STATUS BY TRUE ARCHETYPE (%)")
    print("-" * 90)

    print(
        crosstab.to_string()
    )

    print("\nBEHAVIOURAL CHANGE BY ARCHETYPE (%)")
    print("-" * 90)

    print(
        profile.to_string(index=False)
    )

    print("\nDETERIORATION DETECTION SUMMARY (%)")
    print("-" * 90)

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

    momentum = pd.read_parquet(
        MOMENTUM_FILE
    )

    validation = build_validation_data(
        customer_master,
        momentum,
    )

    crosstab = build_momentum_crosstab(
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