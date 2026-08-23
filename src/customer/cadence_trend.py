from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRANSACTION_FILE = Path(
    "data/generated/transactions.parquet"
)

CUSTOMER_FILE = Path(
    "data/generated/customer_master.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/customer_cadence_trend.parquet"
)

MIN_GAPS_FOR_TREND = 5


# ---------------------------------------------------------------------
# Purchase gap history
# ---------------------------------------------------------------------

def build_purchase_gap_history(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    purchase_dates = (
        transactions[
            [
                "customer_id",
                "transaction_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "customer_id",
                "transaction_date",
            ]
        )
        .copy()
    )

    purchase_dates["previous_purchase_date"] = (
        purchase_dates
        .groupby("customer_id")[
            "transaction_date"
        ]
        .shift(1)
    )

    purchase_dates["purchase_gap_days"] = (
        purchase_dates["transaction_date"]
        - purchase_dates["previous_purchase_date"]
    ).dt.days

    gaps = purchase_dates.dropna(
        subset=["purchase_gap_days"]
    ).copy()

    gaps["gap_sequence"] = (
        gaps
        .groupby("customer_id")
        .cumcount()
        + 1
    )

    return gaps


# ---------------------------------------------------------------------
# Trend calculation
# ---------------------------------------------------------------------

def calculate_customer_trend(
    customer_gaps: pd.DataFrame,
) -> pd.Series:

    gap_count = len(customer_gaps)

    if gap_count < MIN_GAPS_FOR_TREND:
        return pd.Series(
            {
                "trend_gap_count": gap_count,
                "gap_trend_slope": np.nan,
                "gap_trend_pct": np.nan,
                "gap_trend_r_squared": np.nan,
                "gap_trend_p_value": np.nan,
                "early_median_gap": np.nan,
                "recent_median_gap": np.nan,
                "recent_vs_early_gap_pct": np.nan,
            }
        )

    x = customer_gaps[
        "gap_sequence"
    ].to_numpy()

    y = customer_gaps[
        "purchase_gap_days"
    ].to_numpy()

    regression = linregress(
        x,
        y,
    )

    overall_median_gap = np.median(y)

    if overall_median_gap > 0:
        gap_trend_pct = (
            regression.slope
            / overall_median_gap
        )
    else:
        gap_trend_pct = np.nan

    split_index = gap_count // 2

    early_gaps = y[:split_index]
    recent_gaps = y[split_index:]

    early_median = np.median(
        early_gaps
    )

    recent_median = np.median(
        recent_gaps
    )

    if early_median > 0:
        recent_vs_early = (
            recent_median
            - early_median
        ) / early_median
    else:
        recent_vs_early = np.nan

    return pd.Series(
        {
            "trend_gap_count": gap_count,
            "gap_trend_slope":
                regression.slope,
            "gap_trend_pct":
                gap_trend_pct,
            "gap_trend_r_squared":
                regression.rvalue ** 2,
            "gap_trend_p_value":
                regression.pvalue,
            "early_median_gap":
                early_median,
            "recent_median_gap":
                recent_median,
            "recent_vs_early_gap_pct":
                recent_vs_early,
        }
    )


def build_customer_trends(
    gaps: pd.DataFrame,
) -> pd.DataFrame:

    trends = (
        gaps
        .groupby(
            "customer_id",
            group_keys=False,
        )
        .apply(
            calculate_customer_trend,
            include_groups=False,
        )
        .reset_index()
    )

    return trends


# ---------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------

def add_trend_status(
    trends: pd.DataFrame,
) -> pd.DataFrame:

    result = trends.copy()

    sufficient_history = (
        result["trend_gap_count"]
        >= MIN_GAPS_FOR_TREND
    )

    strong_positive_trend = (
        sufficient_history
        & result["gap_trend_pct"].ge(0.03)
        & result["gap_trend_r_squared"].ge(0.20)
        & result["recent_vs_early_gap_pct"].ge(0.25)
    )

    positive_trend = (
        sufficient_history
        & result["gap_trend_pct"].ge(0.015)
        & result["recent_vs_early_gap_pct"].ge(0.15)
    )

    improving_trend = (
        sufficient_history
        & result["gap_trend_pct"].le(-0.015)
        & result["recent_vs_early_gap_pct"].le(-0.15)
    )

    conditions = [
        ~sufficient_history,
        strong_positive_trend,
        positive_trend,
        improving_trend,
    ]

    labels = [
        "Insufficient History",
        "Strongly Deteriorating",
        "Deteriorating",
        "Improving",
    ]

    result["cadence_trend_status"] = (
        np.select(
            conditions,
            labels,
            default="Stable / Mixed",
        )
    )

    return result


# ---------------------------------------------------------------------
# Complete output
# ---------------------------------------------------------------------

def build_output(
    customers: pd.DataFrame,
    trends: pd.DataFrame,
) -> pd.DataFrame:

    output = customers[
        [
            "customer_id",
            "acquisition_date",
        ]
    ].merge(
        trends,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    output[
        "trend_gap_count"
    ] = (
        output["trend_gap_count"]
        .fillna(0)
        .astype(int)
    )

    output[
        "cadence_trend_status"
    ] = (
        output[
            "cadence_trend_status"
        ]
        .fillna(
            "Insufficient History"
        )
    )

    return output


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    trends: pd.DataFrame,
) -> None:

    print("\nCADENCE TREND QA")
    print("=" * 75)

    print(
        f"Customers: "
        f"{len(trends):,}"
    )

    sufficient = (
        trends[
            "trend_gap_count"
        ]
        >= MIN_GAPS_FOR_TREND
    )

    print(
        f"Customers with sufficient trend history: "
        f"{sufficient.sum():,}"
    )

    print("\nCadence trend status:")

    print(
        trends[
            "cadence_trend_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nMedian trend measures "
        "among sufficient-history customers:"
    )

    metrics = [
        "gap_trend_slope",
        "gap_trend_pct",
        "gap_trend_r_squared",
        "recent_vs_early_gap_pct",
    ]

    print(
        trends.loc[
            sufficient,
            metrics,
        ]
        .median()
        .round(3)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading source data..."
    )

    transactions = (
        pd.read_parquet(
            TRANSACTION_FILE
        )
    )

    customers = (
        pd.read_parquet(
            CUSTOMER_FILE
        )
    )

    print(
        "Building purchase gap history..."
    )

    gaps = build_purchase_gap_history(
        transactions
    )

    print(
        "Calculating customer cadence trends..."
    )

    trends = build_customer_trends(
        gaps
    )

    trends = add_trend_status(
        trends
    )

    output = build_output(
        customers,
        trends,
    )

    output.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(output)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()