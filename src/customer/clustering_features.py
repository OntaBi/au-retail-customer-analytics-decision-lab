from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AS_OF_DATE = pd.Timestamp("2026-07-31")

CUSTOMER_FILE = Path(
    "data/generated/customer_master.parquet"
)

TRANSACTION_FILE = Path(
    "data/generated/transactions.parquet"
)

FEATURE_FILE = Path(
    "data/runtime/customer_features.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/clustering_features.parquet"
)


# ---------------------------------------------------------------------
# Core transaction behaviour
# ---------------------------------------------------------------------

def build_core_behaviour(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    core = (
        transactions
        .groupby("customer_id")
        .agg(
            orders=("order_id", "nunique"),
            total_sales=("net_sales", "sum"),
            total_margin=("gross_margin", "sum"),
            total_units=("units", "sum"),
            avg_discount_pct=("discount_pct", "mean"),
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
        )
        .reset_index()
    )

    core["avg_order_value"] = (
        core["total_sales"]
        / core["orders"]
    )

    core["avg_margin_per_order"] = (
        core["total_margin"]
        / core["orders"]
    )

    core["units_per_order"] = (
        core["total_units"]
        / core["orders"]
    )

    core["tenure_days"] = (
        AS_OF_DATE
        - core["first_purchase_date"]
    ).dt.days

    core["days_since_last_purchase"] = (
        AS_OF_DATE
        - core["last_purchase_date"]
    ).dt.days

    return core


# ---------------------------------------------------------------------
# Discount behaviour
# ---------------------------------------------------------------------

def build_discount_behaviour(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    discount = transactions[
        [
            "customer_id",
            "order_id",
            "discount_pct",
        ]
    ].copy()

    discount["discounted_order"] = (
        discount["discount_pct"] > 0
    )

    discount_summary = (
        discount
        .groupby("customer_id")
        .agg(
            discounted_order_share=(
                "discounted_order",
                "mean",
            ),
            avg_discount_pct=(
                "discount_pct",
                "mean",
            ),
            max_discount_pct=(
                "discount_pct",
                "max",
            ),
        )
        .reset_index()
    )

    return discount_summary


# ---------------------------------------------------------------------
# Channel behaviour
# ---------------------------------------------------------------------

def build_channel_behaviour(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    channel_counts = (
        transactions
        .groupby(
            [
                "customer_id",
                "channel",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for channel in [
        "Store",
        "Online",
        "Click & Collect",
    ]:
        if channel not in channel_counts.columns:
            channel_counts[channel] = 0

    total_orders = (
        channel_counts.sum(axis=1)
    )

    channel_features = pd.DataFrame(
        {
            "customer_id":
                channel_counts.index,

            "store_share":
                (
                    channel_counts["Store"]
                    / total_orders
                ),

            "online_share":
                (
                    channel_counts["Online"]
                    / total_orders
                ),

            "click_collect_share":
                (
                    channel_counts["Click & Collect"]
                    / total_orders
                ),

            "channels_used":
                (
                    channel_counts.gt(0)
                    .sum(axis=1)
                ),
        }
    ).reset_index(drop=True)

    return channel_features


# ---------------------------------------------------------------------
# Category behaviour
# ---------------------------------------------------------------------

def build_category_behaviour(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    category_counts = (
        transactions
        .groupby(
            [
                "customer_id",
                "category",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    total_orders = (
        category_counts.sum(axis=1)
    )

    category_share = (
        category_counts
        .div(
            total_orders,
            axis=0,
        )
    )

    # Herfindahl-style concentration:
    # closer to 1 = heavily concentrated in one category.
    category_concentration = (
        category_share
        .pow(2)
        .sum(axis=1)
    )

    dominant_category_share = (
        category_share.max(axis=1)
    )

    categories_used = (
        category_counts
        .gt(0)
        .sum(axis=1)
    )

    category_features = pd.DataFrame(
        {
            "customer_id":
                category_counts.index,

            "categories_used":
                categories_used,

            "category_concentration":
                category_concentration,

            "dominant_category_share":
                dominant_category_share,
        }
    ).reset_index(drop=True)

    return category_features


# ---------------------------------------------------------------------
# Cadence behaviour
# ---------------------------------------------------------------------

def build_cadence_behaviour(
    customer_features: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "customer_id",
        "median_purchase_gap_days",
        "mean_purchase_gap_days",
        "purchase_gap_std_days",
        "cadence_cv",
        "observed_purchase_gaps",
    ]

    return customer_features[
        columns
    ].copy()


# ---------------------------------------------------------------------
# Build clustering feature table
# ---------------------------------------------------------------------

def build_clustering_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:

    # Only customers with at least one purchase are cluster candidates.
    purchasing_customers = (
        customers[
            ["customer_id"]
        ]
        .merge(
            transactions[
                ["customer_id"]
            ]
            .drop_duplicates(),
            on="customer_id",
            how="inner",
        )
    )

    core = build_core_behaviour(
        transactions
    )

    discount = build_discount_behaviour(
        transactions
    )

    channel = build_channel_behaviour(
        transactions
    )

    category = build_category_behaviour(
        transactions
    )

    cadence = build_cadence_behaviour(
        customer_features
    )

    features = (
        purchasing_customers
        .merge(
            core,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            discount,
            on="customer_id",
            how="left",
            validate="one_to_one",
            suffixes=(
                "",
                "_discount",
            ),
        )
        .merge(
            channel,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            category,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            cadence,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    # Keep one average discount field.
    if "avg_discount_pct_discount" in features.columns:
        features["avg_discount_pct"] = (
            features[
                "avg_discount_pct_discount"
            ]
        )

        features = features.drop(
            columns=[
                "avg_discount_pct_discount"
            ]
        )

    return features


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    features: pd.DataFrame,
) -> None:

    print("\nCLUSTERING FEATURE QA")
    print("=" * 80)

    print(
        f"Customers available for clustering: "
        f"{len(features):,}"
    )

    print(
        f"Feature columns: "
        f"{len(features.columns) - 1}"
    )

    print("\nMissing values by feature:")

    missing = (
        features
        .drop(
            columns=["customer_id"]
        )
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(
        missing[
            missing > 0
        ]
    )

    print("\nBehavioural medians:")

    median_columns = [
        "orders",
        "avg_order_value",
        "units_per_order",
        "avg_discount_pct",
        "discounted_order_share",
        "store_share",
        "online_share",
        "channels_used",
        "categories_used",
        "category_concentration",
        "median_purchase_gap_days",
        "cadence_cv",
    ]

    available_columns = [
        column
        for column in median_columns
        if column in features.columns
    ]

    print(
        features[
            available_columns
        ]
        .median(
            numeric_only=True
        )
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

    print("Loading source data...")

    customers = pd.read_parquet(
        CUSTOMER_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    customer_features = pd.read_parquet(
        FEATURE_FILE
    )

    print(
        "Building clustering feature matrix..."
    )

    features = build_clustering_features(
        customers,
        transactions,
        customer_features,
    )

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(features)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()