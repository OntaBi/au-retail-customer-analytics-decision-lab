from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRANSACTION_FILE = Path(
    "data/runtime/golden_customer_transactions.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/cross_sell_training.parquet"
)

AS_OF_DATE = pd.Timestamp("2026-07-31")

OBSERVATION_DATES = pd.to_datetime(
    [
        "2024-01-31",
        "2024-04-30",
        "2024-07-31",
        "2024-10-31",
        "2025-01-31",
        "2025-04-30",
        "2025-07-31",
        "2025-10-31",
        "2026-01-31",
    ]
)

OUTCOME_DAYS = 180
LOOKBACK_180_DAYS = 180
LOOKBACK_365_DAYS = 365

MIN_LIFETIME_ORDERS = 3
MIN_PURCHASE_TENURE_DAYS = 180
MIN_EXISTING_CATEGORIES = 1


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    tx = transactions.copy()

    required = {
        "golden_customer_id",
        "transaction_date",
        "order_id",
        "net_sales",
        "gross_margin",
        "units",
        "discount_pct",
        "channel",
        "category",
    }

    missing = required - set(tx.columns)

    if missing:
        raise ValueError(
            "Golden customer transactions are missing required columns: "
            f"{sorted(missing)}"
        )

    tx["transaction_date"] = pd.to_datetime(
        tx["transaction_date"]
    )

    tx["discounted_order"] = (
        tx["discount_pct"]
        .fillna(0)
        .gt(0)
    )

    return tx


# ---------------------------------------------------------------------
# Historical feature construction
# ---------------------------------------------------------------------

def build_snapshot(
    transactions: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:

    history = transactions.loc[
        transactions["transaction_date"].le(
            observation_date
        )
    ].copy()

    if history.empty:
        return pd.DataFrame()

    base = (
        history
        .groupby("golden_customer_id")
        .agg(
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
            lifetime_orders=("order_id", "nunique"),
            lifetime_sales=("net_sales", "sum"),
            lifetime_margin=("gross_margin", "sum"),
            lifetime_units=("units", "sum"),
            avg_order_value=("net_sales", "mean"),
            avg_margin_per_order=("gross_margin", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            discounted_order_share=("discounted_order", "mean"),
            channels_used=("channel", "nunique"),
            categories_used=("category", "nunique"),
        )
        .reset_index()
    )

    base["days_since_last_purchase"] = (
        observation_date
        - base["last_purchase_date"]
    ).dt.days

    base["purchase_tenure_days"] = (
        observation_date
        - base["first_purchase_date"]
    ).dt.days

    base["margin_rate"] = safe_divide(
        base["lifetime_margin"],
        base["lifetime_sales"],
    )

    start_180 = (
        observation_date
        - pd.Timedelta(
            days=LOOKBACK_180_DAYS - 1
        )
    )

    start_365 = (
        observation_date
        - pd.Timedelta(
            days=LOOKBACK_365_DAYS - 1
        )
    )

    recent_180 = history.loc[
        history["transaction_date"].between(
            start_180,
            observation_date,
        )
    ]

    recent_365 = history.loc[
        history["transaction_date"].between(
            start_365,
            observation_date,
        )
    ]

    agg_180 = (
        recent_180
        .groupby("golden_customer_id")
        .agg(
            orders_prior_180d=("order_id", "nunique"),
            sales_prior_180d=("net_sales", "sum"),
            margin_prior_180d=("gross_margin", "sum"),
            categories_prior_180d=("category", "nunique"),
            channels_prior_180d=("channel", "nunique"),
        )
        .reset_index()
    )

    agg_365 = (
        recent_365
        .groupby("golden_customer_id")
        .agg(
            orders_prior_365d=("order_id", "nunique"),
            sales_prior_365d=("net_sales", "sum"),
            margin_prior_365d=("gross_margin", "sum"),
            categories_prior_365d=("category", "nunique"),
            channels_prior_365d=("channel", "nunique"),
        )
        .reset_index()
    )

    base = (
        base
        .merge(
            agg_180,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            agg_365,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    for column in [
        "orders_prior_180d",
        "sales_prior_180d",
        "margin_prior_180d",
        "categories_prior_180d",
        "channels_prior_180d",
        "orders_prior_365d",
        "sales_prior_365d",
        "margin_prior_365d",
        "categories_prior_365d",
        "channels_prior_365d",
    ]:
        base[column] = (
            base[column]
            .fillna(0)
        )

    category_orders = (
        history
        .groupby(
            [
                "golden_customer_id",
                "category",
            ]
        )
        .agg(
            category_orders=("order_id", "nunique"),
            category_sales=("net_sales", "sum"),
            category_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    category_totals = (
        category_orders
        .groupby("golden_customer_id")
        .agg(
            total_category_orders=("category_orders", "sum"),
            total_category_sales=("category_sales", "sum"),
            total_category_margin=("category_margin", "sum"),
        )
        .reset_index()
    )

    category_orders = category_orders.merge(
        category_totals,
        on="golden_customer_id",
        how="left",
        validate="many_to_one",
    )

    category_orders["category_order_share"] = safe_divide(
        category_orders["category_orders"],
        category_orders["total_category_orders"],
    )

    category_orders["category_sales_share"] = safe_divide(
        category_orders["category_sales"],
        category_orders["total_category_sales"],
    )

    category_profile = (
        category_orders
        .groupby("golden_customer_id")
        .agg(
            dominant_category_share=(
                "category_order_share",
                "max",
            ),
            dominant_category_sales_share=(
                "category_sales_share",
                "max",
            ),
        )
        .reset_index()
    )

    category_hhi = (
        category_orders
        .assign(
            squared_share=lambda x:
                x["category_order_share"] ** 2
        )
        .groupby("golden_customer_id")
        .agg(
            category_concentration=(
                "squared_share",
                "sum",
            )
        )
        .reset_index()
    )

    base = (
        base
        .merge(
            category_profile,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            category_hhi,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    channel_orders = (
        history
        .groupby(
            [
                "golden_customer_id",
                "channel",
            ]
        )
        .agg(
            channel_orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    channel_orders["total_orders"] = (
        channel_orders
        .groupby("golden_customer_id")[
            "channel_orders"
        ]
        .transform("sum")
    )

    channel_orders["channel_share"] = safe_divide(
        channel_orders["channel_orders"],
        channel_orders["total_orders"],
    )

    channel_profile = (
        channel_orders
        .groupby("golden_customer_id")
        .agg(
            preferred_channel_share=(
                "channel_share",
                "max",
            )
        )
        .reset_index()
    )

    base = base.merge(
        channel_profile,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    all_categories = sorted(
        transactions["category"]
        .dropna()
        .unique()
        .tolist()
    )

    total_categories = len(
        all_categories
    )

    base["category_whitespace_count"] = (
        total_categories
        - base["categories_used"]
    )

    base["category_coverage_pct"] = (
        base["categories_used"]
        / float(total_categories)
        if total_categories > 0
        else np.nan
    )

    base["cross_sell_eligible"] = (
        base["lifetime_orders"]
        .ge(MIN_LIFETIME_ORDERS)
        & base["purchase_tenure_days"]
        .ge(MIN_PURCHASE_TENURE_DAYS)
        & base["categories_used"]
        .ge(MIN_EXISTING_CATEGORIES)
        & base["category_whitespace_count"]
        .gt(0)
    )

    base["observation_date"] = (
        observation_date
    )

    return base


# ---------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------

def add_cross_sell_outcome(
    snapshot: pd.DataFrame,
    transactions: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:

    history = transactions.loc[
        transactions["transaction_date"].le(
            observation_date
        )
    ].copy()

    outcome_end = (
        observation_date
        + pd.Timedelta(
            days=OUTCOME_DAYS
        )
    )

    future = transactions.loc[
        transactions["transaction_date"].gt(
            observation_date
        )
        & transactions["transaction_date"].le(
            outcome_end
        )
    ].copy()

    prior_categories = (
        history
        .groupby("golden_customer_id")[
            "category"
        ]
        .agg(
            lambda s:
                set(
                    s.dropna()
                )
        )
        .to_dict()
    )

    future_categories = (
        future
        .groupby("golden_customer_id")[
            "category"
        ]
        .agg(
            lambda s:
                set(
                    s.dropna()
                )
        )
        .to_dict()
    )

    new_category_counts = []
    entered_new_category = []

    for customer_id in snapshot[
        "golden_customer_id"
    ]:
        prior = prior_categories.get(
            customer_id,
            set(),
        )
        future_set = future_categories.get(
            customer_id,
            set(),
        )
        new_categories = (
            future_set
            - prior
        )

        new_category_counts.append(
            len(
                new_categories
            )
        )

        entered_new_category.append(
            int(
                len(
                    new_categories
                )
                > 0
            )
        )

    result = snapshot.copy()

    result[
        "new_categories_next_180d"
    ] = new_category_counts

    result[
        "entered_new_category_next_180d"
    ] = entered_new_category

    future_summary = (
        future
        .groupby("golden_customer_id")
        .agg(
            future_180d_orders=("order_id", "nunique"),
            future_180d_sales=("net_sales", "sum"),
            future_180d_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    result = result.merge(
        future_summary,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "future_180d_orders",
        "future_180d_sales",
        "future_180d_margin",
    ]:
        result[column] = (
            result[column]
            .fillna(0)
        )

    return result


# ---------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------

def build_training_dataset(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    max_transaction_date = (
        transactions["transaction_date"]
        .max()
    )

    for observation_date in OBSERVATION_DATES:
        outcome_end = (
            observation_date
            + pd.Timedelta(
                days=OUTCOME_DAYS
            )
        )

        if outcome_end > max_transaction_date:
            continue

        snapshot = build_snapshot(
            transactions,
            observation_date,
        )

        if snapshot.empty:
            continue

        snapshot = add_cross_sell_outcome(
            snapshot,
            transactions,
            observation_date,
        )

        snapshot = snapshot.loc[
            snapshot["cross_sell_eligible"]
        ].copy()

        rows.append(
            snapshot
        )

    if not rows:
        return pd.DataFrame()

    training = pd.concat(
        rows,
        ignore_index=True,
    )

    return (
        training
        .sort_values(
            [
                "observation_date",
                "golden_customer_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    training: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("CROSS-SELL PROPENSITY TRAINING QA")
    print("=" * 78)

    if training.empty:
        print(
            "No training rows created."
        )
        return

    positive_rate = (
        training[
            "entered_new_category_next_180d"
        ]
        .mean()
    )

    print(
        f"Training rows                    : "
        f"{len(training):,}"
    )

    print(
        f"Unique golden customers          : "
        f"{training['golden_customer_id'].nunique():,}"
    )

    print(
        f"Observation dates                : "
        f"{training['observation_date'].nunique():,}"
    )

    print(
        f"Observation range                : "
        f"{training['observation_date'].min().date()} "
        f"to "
        f"{training['observation_date'].max().date()}"
    )

    print(
        f"Entered new category next 180d   : "
        f"{positive_rate:.1%}"
    )

    print(
        f"Median categories already used   : "
        f"{training['categories_used'].median():.0f}"
    )

    print(
        f"Median category whitespace       : "
        f"{training['category_whitespace_count'].median():.0f}"
    )

    print(
        f"Median lifetime orders           : "
        f"{training['lifetime_orders'].median():.0f}"
    )

    print(
        f"Eligibility rule                 : "
        f">={MIN_LIFETIME_ORDERS} lifetime orders, "
        f">={MIN_PURCHASE_TENURE_DAYS} tenure days, "
        f">={MIN_EXISTING_CATEGORIES} existing category, "
        f"at least 1 whitespace category"
    )

    print()
    print(
        "Rows by observation date:"
    )

    by_snapshot = (
        training
        .groupby("observation_date")
        .agg(
            rows=("golden_customer_id", "size"),
            customers=("golden_customer_id", "nunique"),
            positive_rate=(
                "entered_new_category_next_180d",
                "mean",
            ),
            median_categories_used=(
                "categories_used",
                "median",
            ),
            median_whitespace=(
                "category_whitespace_count",
                "median",
            ),
        )
    )

    by_snapshot[
        "positive_rate"
    ] = (
        by_snapshot[
            "positive_rate"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )

    print(
        by_snapshot.to_string()
    )

    print()
    print(
        "Leakage guard outcome fields: "
        "new_categories_next_180d, "
        "future_180d_orders, "
        "future_180d_sales, "
        "future_180d_margin, "
        "entered_new_category_next_180d"
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
        "Loading golden customer transactions..."
    )

    transactions = prepare_transactions(
        pd.read_parquet(
            TRANSACTION_FILE
        )
    )

    print(
        "Building historical cross-sell snapshots..."
    )

    training = build_training_dataset(
        transactions
    )

    if training.empty:
        raise ValueError(
            "No cross-sell training rows were created."
        )

    training.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(
        training
    )

    print()
    print(
        "File created:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
