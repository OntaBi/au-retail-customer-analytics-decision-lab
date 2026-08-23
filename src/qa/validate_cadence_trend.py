from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CUSTOMER_MASTER_FILE = Path(
    "data/generated/customer_master.parquet"
)

TREND_FILE = Path(
    "data/runtime/customer_cadence_trend.parquet"
)

OUTPUT_DIR = Path("outputs")

CROSSTAB_FILE = OUTPUT_DIR / "cadence_trend_crosstab.csv"
PROFILE_FILE = OUTPUT_DIR / "cadence_trend_profile.csv"


# ---------------------------------------------------------------------
# Validation data
# ---------------------------------------------------------------------

def build_validation_data(
    customer_master: pd.DataFrame,
    cadence_trend: pd.DataFrame,
) -> pd.DataFrame:

    truth = customer_master[
        [
            "customer_id",
            "archetype",
        ]
    ].copy()

    return truth.merge(
        cadence_trend,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )


# ---------------------------------------------------------------------
# Cross-tab
# ---------------------------------------------------------------------

def build_crosstab(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["archetype"],
        validation["cadence_trend_status"],
        normalize="index",
    )

    return (
        crosstab
        .mul(100)
        .round(1)
    )


# ---------------------------------------------------------------------
# Trend profile
# ---------------------------------------------------------------------

def build_profile(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    sufficient = validation.loc[
        validation["trend_gap_count"] >= 5
    ].copy()

    profile = (
        sufficient
        .groupby("archetype")
        .agg(
            customers=("customer_id", "nunique"),
            median_gap_count=(
                "trend_gap_count",
                "median",
            ),
            median_slope=(
                "gap_trend_slope",
                "median",
            ),
            median_trend_pct=(
                "gap_trend_pct",
                "median",
            ),
            median_r_squared=(
                "gap_trend_r_squared",
                "median",
            ),
            median_recent_vs_early=(
                "recent_vs_early_gap_pct",
                "median",
            ),
        )
        .reset_index()
    )

    profile[
        [
            "median_slope",
            "median_trend_pct",
            "median_r_squared",
            "median_recent_vs_early",
        ]
    ] = (
        profile[
            [
                "median_slope",
                "median_trend_pct",
                "median_r_squared",
                "median_recent_vs_early",
            ]
        ]
        .round(3)
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
        result["cadence_trend_status"]
        .isin(
            [
                "Deteriorating",
                "Strongly Deteriorating",
            ]
        )
    )

    result["strong_detected"] = (
        result["cadence_trend_status"]
        .eq("Strongly Deteriorating")
    )

    result["sufficient_history"] = (
        result["trend_gap_count"] >= 5
    )

    overall = (
        result
        .groupby("archetype")
        .agg(
            customers=("customer_id", "nunique"),
            sufficient_history_rate=(
                "sufficient_history",
                "mean",
            ),
            deterioration_rate=(
                "deterioration_detected",
                "mean",
            ),
            strong_rate=(
                "strong_detected",
                "mean",
            ),
        )
        .reset_index()
    )

    sufficient = result.loc[
        result["sufficient_history"]
    ].copy()

    sufficient_summary = (
        sufficient
        .groupby("archetype")
        .agg(
            sufficient_customers=(
                "customer_id",
                "nunique",
            ),
            sufficient_deterioration_rate=(
                "deterioration_detected",
                "mean",
            ),
            sufficient_strong_rate=(
                "strong_detected",
                "mean",
            ),
        )
        .reset_index()
    )

    summary = overall.merge(
        sufficient_summary,
        on="archetype",
        how="left",
    )

    rate_columns = [
        "sufficient_history_rate",
        "deterioration_rate",
        "strong_rate",
        "sufficient_deterioration_rate",
        "sufficient_strong_rate",
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

def print_results(
    validation: pd.DataFrame,
    crosstab: pd.DataFrame,
    profile: pd.DataFrame,
    detection: pd.DataFrame,
) -> None:

    print("\nCADENCE TREND VALIDATION")
    print("=" * 90)

    print(
        f"Customers validated: "
        f"{validation['customer_id'].nunique():,}"
    )

    print("\nCADENCE TREND STATUS BY TRUE ARCHETYPE (%)")
    print("-" * 90)

    print(crosstab.to_string())

    print("\nTREND PROFILE BY ARCHETYPE")
    print("-" * 90)

    print(
        profile.to_string(index=False)
    )

    print("\nDETERIORATION DETECTION SUMMARY (%)")
    print("-" * 90)

    print(
        detection.to_string(index=False)
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

    cadence_trend = pd.read_parquet(
        TREND_FILE
    )

    validation = build_validation_data(
        customer_master,
        cadence_trend,
    )

    crosstab = build_crosstab(
        validation
    )

    profile = build_profile(
        validation
    )

    detection = build_detection_summary(
        validation
    )

    crosstab.to_csv(
        CROSSTAB_FILE
    )

    profile.to_csv(
        PROFILE_FILE,
        index=False,
    )

    print_results(
        validation,
        crosstab,
        profile,
        detection,
    )

    print("\nFiles created:")
    print(CROSSTAB_FILE)
    print(PROFILE_FILE)


if __name__ == "__main__":
    main()