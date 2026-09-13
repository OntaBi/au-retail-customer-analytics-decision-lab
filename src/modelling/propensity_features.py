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

GOLDEN_MASTER_FILE = Path(
    "data/runtime/golden_customer_master.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/reengagement_training.parquet"
)

AS_OF_DATE = pd.Timestamp("2026-07-31")

# Historical observation dates used to construct training snapshots.
# These are deliberately month-end dates and stop early enough to allow
# a complete 90-day future outcome window.
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
        "2026-04-30",
    ]
)

OUTCOME_DAYS = 90
LOOKBACK_180_DAYS = 180
LOOKBACK_365_DAYS = 365

MIN_PRIOR_ORDERS_FOR_CADENCE = 3
MIN_PRIOR_GAPS_FOR_STABLE_CADENCE = 2
MIN_PURCHASE_TENURE_DAYS = 180

# Re-engagement eligibility should represent a customer who has
# meaningfully broken their established purchase pattern, rather than
# someone who is merely between normal purchases.
MIN_DAYS_SINCE_PURCHASE = 45
MIN_LAPSE_RATIO = 1.50


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def month_end(date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(date) + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------
# Transaction preparation
# ---------------------------------------------------------------------

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

def build_purchase_dates(
    history: pd.DataFrame,
) -> pd.DataFrame:

    purchase_dates = (
        history[
            [
                "golden_customer_id",
                "transaction_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "golden_customer_id",
                "transaction_date",
            ]
        )
        .copy()
    )

    purchase_dates["previous_purchase_date"] = (
        purchase_dates
        .groupby("golden_customer_id")[
            "transaction_date"
        ]
        .shift(1)
    )

    purchase_dates["purchase_gap_days"] = (
        purchase_dates["transaction_date"]
        - purchase_dates["previous_purchase_date"]
    ).dt.days

    return purchase_dates


def build_customer_history_features(
    history: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:

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

    # --------------------------------------------------------------
    # Recent-window activity
    # --------------------------------------------------------------

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
    ].copy()

    recent_365 = history.loc[
        history["transaction_date"].between(
            start_365,
            observation_date,
        )
    ].copy()

    agg_180 = (
        recent_180
        .groupby("golden_customer_id")
        .agg(
            orders_prior_180d=("order_id", "nunique"),
            sales_prior_180d=("net_sales", "sum"),
            margin_prior_180d=("gross_margin", "sum"),
            units_prior_180d=("units", "sum"),
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
            units_prior_365d=("units", "sum"),
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
        "units_prior_180d",
        "orders_prior_365d",
        "sales_prior_365d",
        "margin_prior_365d",
        "units_prior_365d",
    ]:
        base[column] = (
            base[column]
            .fillna(0)
        )

    # --------------------------------------------------------------
    # Historical cadence
    # --------------------------------------------------------------

    purchase_dates = build_purchase_dates(
        history
    )

    gap_summary = (
        purchase_dates
        .groupby("golden_customer_id")
        .agg(
            observed_purchase_gaps=(
                "purchase_gap_days",
                lambda s:
                    s.notna().sum(),
            ),
            median_purchase_gap_days=(
                "purchase_gap_days",
                "median",
            ),
            mean_purchase_gap_days=(
                "purchase_gap_days",
                "mean",
            ),
            std_purchase_gap_days=(
                "purchase_gap_days",
                "std",
            ),
        )
        .reset_index()
    )

    base = base.merge(
        gap_summary,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    base["cadence_cv"] = safe_divide(
        base["std_purchase_gap_days"],
        base["mean_purchase_gap_days"],
    )

    base["expected_return_days"] = (
        base["median_purchase_gap_days"]
    )

    base["adjusted_lapse_ratio"] = safe_divide(
        base["days_since_last_purchase"],
        base["expected_return_days"],
    )

    # --------------------------------------------------------------
    # Prior vs recent momentum within history only
    # --------------------------------------------------------------

    recent_window_start = (
        observation_date
        - pd.Timedelta(days=179)
    )

    prior_window_end = (
        recent_window_start
        - pd.Timedelta(days=1)
    )

    prior_window_start = (
        prior_window_end
        - pd.Timedelta(days=179)
    )

    prior = history.loc[
        history["transaction_date"].between(
            prior_window_start,
            prior_window_end,
        )
    ]

    recent = history.loc[
        history["transaction_date"].between(
            recent_window_start,
            observation_date,
        )
    ]

    prior_agg = (
        prior
        .groupby("golden_customer_id")
        .agg(
            prior_orders=("order_id", "nunique"),
            prior_sales=("net_sales", "sum"),
            prior_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    recent_agg = (
        recent
        .groupby("golden_customer_id")
        .agg(
            recent_orders=("order_id", "nunique"),
            recent_sales=("net_sales", "sum"),
            recent_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    base = (
        base
        .merge(
            prior_agg,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            recent_agg,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    for column in [
        "prior_orders",
        "prior_sales",
        "prior_margin",
        "recent_orders",
        "recent_sales",
        "recent_margin",
    ]:
        base[column] = (
            base[column]
            .fillna(0)
        )

    base["order_change_pct"] = safe_divide(
        base["recent_orders"] - base["prior_orders"],
        base["prior_orders"],
    )

    base["sales_change_pct"] = safe_divide(
        base["recent_sales"] - base["prior_sales"],
        base["prior_sales"],
    )

    base["margin_change_pct"] = safe_divide(
        base["recent_margin"] - base["prior_margin"],
        base["prior_margin"],
    )

    # --------------------------------------------------------------
    # Category concentration
    # --------------------------------------------------------------

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
        )
        .reset_index()
    )

    category_orders[
        "total_category_orders"
    ] = (
        category_orders
        .groupby("golden_customer_id")[
            "category_orders"
        ]
        .transform("sum")
    )

    category_orders[
        "category_order_share"
    ] = safe_divide(
        category_orders["category_orders"],
        category_orders["total_category_orders"],
    )

    category_profile = (
        category_orders
        .groupby("golden_customer_id")
        .agg(
            dominant_category_share=(
                "category_order_share",
                "max",
            ),
        )
        .reset_index()
    )

    base = base.merge(
        category_profile,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------------
    # Channel concentration
    # --------------------------------------------------------------

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

    channel_orders[
        "total_channel_orders"
    ] = (
        channel_orders
        .groupby("golden_customer_id")[
            "channel_orders"
        ]
        .transform("sum")
    )

    channel_orders[
        "channel_order_share"
    ] = safe_divide(
        channel_orders["channel_orders"],
        channel_orders["total_channel_orders"],
    )

    channel_profile = (
        channel_orders
        .groupby("golden_customer_id")
        .agg(
            preferred_channel_share=(
                "channel_order_share",
                "max",
            ),
        )
        .reset_index()
    )

    base = base.merge(
        channel_profile,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------------

    base["cadence_history_sufficient"] = (
        base["lifetime_orders"]
        .ge(MIN_PRIOR_ORDERS_FOR_CADENCE)
        & base["observed_purchase_gaps"]
        .fillna(0)
        .ge(MIN_PRIOR_GAPS_FOR_STABLE_CADENCE)
        & base["purchase_tenure_days"]
        .ge(MIN_PURCHASE_TENURE_DAYS)
    )

    base["reengagement_eligible"] = (
        base["cadence_history_sufficient"]
        & base["days_since_last_purchase"]
        .ge(MIN_DAYS_SINCE_PURCHASE)
        & base["adjusted_lapse_ratio"]
        .fillna(0)
        .ge(MIN_LAPSE_RATIO)
    )

    base["observation_date"] = (
        observation_date
    )

    return base


# ---------------------------------------------------------------------
# Outcome construction
# ---------------------------------------------------------------------

def add_reengagement_outcome(
    snapshot: pd.DataFrame,
    all_transactions: pd.DataFrame,
    observation_date: pd.Timestamp,
) -> pd.DataFrame:

    outcome_end = (
        observation_date
        + pd.Timedelta(
            days=OUTCOME_DAYS
        )
    )

    future = all_transactions.loc[
        all_transactions[
            "transaction_date"
        ].gt(observation_date)
        & all_transactions[
            "transaction_date"
        ].le(outcome_end)
    ].copy()

    future_summary = (
        future
        .groupby("golden_customer_id")
        .agg(
            future_90d_orders=("order_id", "nunique"),
            future_90d_sales=("net_sales", "sum"),
            future_90d_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    result = snapshot.merge(
        future_summary,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "future_90d_orders",
        "future_90d_sales",
        "future_90d_margin",
    ]:
        result[column] = (
            result[column]
            .fillna(0)
        )

    result["purchased_next_90d"] = (
        result["future_90d_orders"]
        .gt(0)
        .astype(int)
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

        history = transactions.loc[
            transactions[
                "transaction_date"
            ].le(observation_date)
        ].copy()

        snapshot = (
            build_customer_history_features(
                history,
                observation_date,
            )
        )

        if snapshot.empty:
            continue

        snapshot = add_reengagement_outcome(
            snapshot,
            transactions,
            observation_date,
        )

        snapshot = snapshot.loc[
            snapshot[
                "reengagement_eligible"
            ]
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

    training = training.sort_values(
        [
            "observation_date",
            "golden_customer_id",
        ]
    ).reset_index(drop=True)

    return training


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    training: pd.DataFrame,
) -> None:

    print("\n90-DAY RETURN PROPENSITY TRAINING QA")
    print("=" * 78)

    if training.empty:
        print("No training rows created.")
        return

    print(
        f"Training rows                 : "
        f"{len(training):,}"
    )

    print(
        f"Unique golden customers       : "
        f"{training['golden_customer_id'].nunique():,}"
    )

    print(
        f"Observation dates             : "
        f"{training['observation_date'].nunique():,}"
    )

    print(
        f"Observation range             : "
        f"{training['observation_date'].min().date()} "
        f"to "
        f"{training['observation_date'].max().date()}"
    )

    positive_rate = (
        training[
            "purchased_next_90d"
        ].mean()
    )

    print(
        f"Purchased next 90d rate       : "
        f"{positive_rate:.1%}"
    )

    print(
        f"Median days since purchase    : "
        f"{training['days_since_last_purchase'].median():.0f}"
    )

    print(
        f"Median lapse ratio            : "
        f"{training['adjusted_lapse_ratio'].median():.2f}"
    )

    print(
        f"Rows with cadence estimate    : "
        f"{training['median_purchase_gap_days'].notna().sum():,}"
    )

    print(
        f"Eligibility rule              : "
        f">={MIN_PRIOR_ORDERS_FOR_CADENCE} prior orders, "
        f">={MIN_PRIOR_GAPS_FOR_STABLE_CADENCE} observed gaps, "
        f">={MIN_PURCHASE_TENURE_DAYS} tenure days, "
        f">={MIN_DAYS_SINCE_PURCHASE} days since purchase, "
        f"lapse ratio >= {MIN_LAPSE_RATIO:.2f}x"
    )

    print("\nRows by observation date:")

    by_snapshot = (
        training
        .groupby("observation_date")
        .agg(
            rows=("golden_customer_id", "size"),
            customers=("golden_customer_id", "nunique"),
            positive_rate=("purchased_next_90d", "mean"),
            median_days_since=(
                "days_since_last_purchase",
                "median",
            ),
            median_lapse_ratio=(
                "adjusted_lapse_ratio",
                "median",
            ),
        )
    )

    by_snapshot["positive_rate"] = (
        by_snapshot["positive_rate"]
        .map(lambda x: f"{x:.1%}")
    )

    print(
        by_snapshot.to_string()
    )

    # Explicit leakage guards.
    forbidden_feature_columns = {
        "future_90d_orders",
        "future_90d_sales",
        "future_90d_margin",
    }

    print(
        "\nLeakage guard target fields present only as outcomes: "
        f"{sorted(forbidden_feature_columns)}"
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

    # The golden master is loaded for reconciliation only. The training
    # set itself is built entirely from observable transaction history.
    golden_master = pd.read_parquet(
        GOLDEN_MASTER_FILE
    )

    if (
        golden_master[
            "golden_customer_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Golden customer master must contain one row per "
            "golden_customer_id."
        )

    print(
        "Building historical re-engagement snapshots..."
    )

    training = build_training_dataset(
        transactions
    )

    if training.empty:
        raise ValueError(
            "No re-engagement training rows were created."
        )

    training.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(
        training
    )

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
