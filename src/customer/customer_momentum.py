from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AS_OF_DATE = pd.Timestamp("2026-07-31")

WINDOW_DAYS = 180

RUNTIME_DIR = Path("data/runtime")

CUSTOMER_FILE = RUNTIME_DIR / "golden_customer_master.parquet"
TRANSACTION_FILE = RUNTIME_DIR / "golden_customer_transactions.parquet"
FEATURE_FILE = RUNTIME_DIR / "customer_features.parquet"

OUTPUT_FILE = RUNTIME_DIR / "customer_momentum.parquet"


# ---------------------------------------------------------------------
# Window definitions
# ---------------------------------------------------------------------

def get_window_dates() -> dict:

    recent_end = AS_OF_DATE
    recent_start = (
        recent_end
        - pd.Timedelta(days=WINDOW_DAYS - 1)
    )

    prior_end = (
        recent_start
        - pd.Timedelta(days=1)
    )

    prior_start = (
        prior_end
        - pd.Timedelta(days=WINDOW_DAYS - 1)
    )

    return {
        "prior_start": prior_start,
        "prior_end": prior_end,
        "recent_start": recent_start,
        "recent_end": recent_end,
    }


# ---------------------------------------------------------------------
# Period metrics
# ---------------------------------------------------------------------

def build_period_metrics(
    transactions: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prefix: str,
) -> pd.DataFrame:

    period = transactions.loc[
        transactions["transaction_date"].between(
            start_date,
            end_date,
            inclusive="both",
        )
    ].copy()

    metrics = (
        period
        .groupby("golden_customer_id")
        .agg(
            orders=("order_id", "nunique"),
            sales=("net_sales", "sum"),
            margin=("gross_margin", "sum"),
            units=("units", "sum"),
            first_purchase=("transaction_date", "min"),
            last_purchase=("transaction_date", "max"),
        )
        .reset_index()
    )

    metrics["aov"] = (
        metrics["sales"]
        / metrics["orders"]
    )

    rename_columns = {
        column: f"{prefix}_{column}"
        for column in metrics.columns
        if column != "golden_customer_id"
    }

    return metrics.rename(
        columns=rename_columns
    )


# ---------------------------------------------------------------------
# Period-specific purchase cadence
# ---------------------------------------------------------------------

def build_period_cadence(
    transactions: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prefix: str,
) -> pd.DataFrame:

    period = (
        transactions.loc[
            transactions["transaction_date"].between(
                start_date,
                end_date,
                inclusive="both",
            ),
            [
                "golden_customer_id",
                "transaction_date",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            [
                "golden_customer_id",
                "transaction_date",
            ]
        )
    )

    period["previous_purchase"] = (
        period
        .groupby("golden_customer_id")[
            "transaction_date"
        ]
        .shift(1)
    )

    period["gap_days"] = (
        period["transaction_date"]
        - period["previous_purchase"]
    ).dt.days

    cadence = (
        period
        .dropna(subset=["gap_days"])
        .groupby("golden_customer_id")
        .agg(
            median_gap_days=(
                "gap_days",
                "median",
            ),
            observed_gaps=(
                "gap_days",
                "count",
            ),
        )
        .reset_index()
    )

    cadence = cadence.rename(
        columns={
            "median_gap_days":
                f"{prefix}_median_gap_days",
            "observed_gaps":
                f"{prefix}_observed_gaps",
        }
    )

    return cadence


# ---------------------------------------------------------------------
# Safe percentage change
# ---------------------------------------------------------------------

def safe_pct_change(
    recent: pd.Series,
    prior: pd.Series,
) -> pd.Series:

    result = np.where(
        prior > 0,
        (recent - prior) / prior,
        np.nan,
    )

    return pd.Series(
        result,
        index=recent.index,
    )


# ---------------------------------------------------------------------
# Build momentum features
# ---------------------------------------------------------------------

def build_momentum_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:

    windows = get_window_dates()

    prior = build_period_metrics(
        transactions,
        windows["prior_start"],
        windows["prior_end"],
        prefix="prior",
    )

    recent = build_period_metrics(
        transactions,
        windows["recent_start"],
        windows["recent_end"],
        prefix="recent",
    )

    prior_cadence = build_period_cadence(
        transactions,
        windows["prior_start"],
        windows["prior_end"],
        prefix="prior",
    )

    recent_cadence = build_period_cadence(
        transactions,
        windows["recent_start"],
        windows["recent_end"],
        prefix="recent",
    )

    momentum = customers[
        [
            "golden_customer_id",
        ]
    ].copy()

    momentum = momentum.merge(
        prior,
        on="golden_customer_id",
        how="left",
    )

    momentum = momentum.merge(
        recent,
        on="golden_customer_id",
        how="left",
    )

    momentum = momentum.merge(
        prior_cadence,
        on="golden_customer_id",
        how="left",
    )

    momentum = momentum.merge(
        recent_cadence,
        on="golden_customer_id",
        how="left",
    )

    # Zero is meaningful for order/sales activity.
    activity_columns = [
        "prior_orders",
        "prior_sales",
        "prior_margin",
        "prior_units",
        "recent_orders",
        "recent_sales",
        "recent_margin",
        "recent_units",
    ]

    momentum[activity_columns] = (
        momentum[activity_columns]
        .fillna(0)
    )

    momentum["order_change_pct"] = safe_pct_change(
        momentum["recent_orders"],
        momentum["prior_orders"],
    )

    momentum["sales_change_pct"] = safe_pct_change(
        momentum["recent_sales"],
        momentum["prior_sales"],
    )

    momentum["margin_change_pct"] = safe_pct_change(
        momentum["recent_margin"],
        momentum["prior_margin"],
    )

    momentum["aov_change_pct"] = safe_pct_change(
        momentum["recent_aov"],
        momentum["prior_aov"],
    )

    # Positive = customer is taking longer to return.
    momentum["cadence_change_pct"] = safe_pct_change(
        momentum["recent_median_gap_days"],
        momentum["prior_median_gap_days"],
    )

    # Comparable history is now based on observable behaviour:
    # the golden customer must have at least one purchase on or before
    # the end of the prior comparison window. This avoids relying on
    # hidden synthetic acquisition dates.
    first_purchase = (
        transactions
        .groupby("golden_customer_id")
        .agg(
            first_observed_purchase_date=(
                "transaction_date",
                "min",
            )
        )
        .reset_index()
    )

    momentum = momentum.merge(
        first_purchase,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    momentum["comparable_history"] = (
        momentum["first_observed_purchase_date"]
        .notna()
        & (
            momentum["first_observed_purchase_date"]
            <= windows["prior_end"]
        )
    )

    # Stronger evidence requirement for cadence comparison.
    momentum["cadence_comparable"] = (
        momentum["comparable_history"]
        & (
            momentum["prior_observed_gaps"]
            .fillna(0)
            >= 2
        )
        & (
            momentum["recent_observed_gaps"]
            .fillna(0)
            >= 2
        )
    )

    feature_columns = customer_features[
        [
            "golden_customer_id",
            "active_customer",
            "lifecycle_status",
            "days_since_last_purchase",
            "adjusted_lapse_ratio",
            "cadence_confidence",
            "trailing_12m_sales",
            "trailing_12m_margin",
        ]
    ]

    momentum = momentum.merge(
        feature_columns,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    return momentum


# ---------------------------------------------------------------------
# Momentum classification
# ---------------------------------------------------------------------

def add_momentum_status(
    momentum: pd.DataFrame,
) -> pd.DataFrame:

    result = momentum.copy()

    insufficient = (
        ~result["comparable_history"]
        | result["prior_orders"].eq(0)
    )

    no_recent_purchase = (
        result["prior_orders"].gt(0)
        & result["recent_orders"].eq(0)
    )

    cadence_deteriorating = (
        result["cadence_comparable"]
        & result["cadence_change_pct"].ge(0.25)
    )

    cadence_strongly_deteriorating = (
        result["cadence_comparable"]
        & result["cadence_change_pct"].ge(0.50)
    )

    orders_deteriorating = (
        result["order_change_pct"].le(-0.25)
    )

    orders_strongly_deteriorating = (
        result["order_change_pct"].le(-0.50)
    )

    sales_deteriorating = (
        result["sales_change_pct"].le(-0.25)
    )

    sales_strongly_deteriorating = (
        result["sales_change_pct"].le(-0.50)
    )

    # Count independent deterioration signals.
    result["momentum_risk_signals"] = (
        cadence_deteriorating.astype(int)
        + orders_deteriorating.astype(int)
        + sales_deteriorating.astype(int)
    )

    result["strong_momentum_risk_signals"] = (
        cadence_strongly_deteriorating.astype(int)
        + orders_strongly_deteriorating.astype(int)
        + sales_strongly_deteriorating.astype(int)
    )

    conditions = [
        insufficient,
        no_recent_purchase,
        (
            result["strong_momentum_risk_signals"]
            >= 2
        ),
        (
            result["momentum_risk_signals"]
            >= 2
        ),
        (
            result["order_change_pct"].ge(0.25)
            & result["sales_change_pct"].ge(0.25)
        ),
    ]

    labels = [
        "Insufficient History",
        "No Recent Purchase",
        "Strongly Deteriorating",
        "Deteriorating",
        "Improving",
    ]

    result["momentum_status"] = np.select(
        conditions,
        labels,
        default="Stable / Mixed",
    )

    return result


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    momentum: pd.DataFrame,
) -> None:

    windows = get_window_dates()

    print("\nCUSTOMER MOMENTUM QA")
    print("=" * 70)

    print(
        f"Prior window:  "
        f"{windows['prior_start'].date()} "
        f"to {windows['prior_end'].date()}"
    )

    print(
        f"Recent window: "
        f"{windows['recent_start'].date()} "
        f"to {windows['recent_end'].date()}"
    )

    print(
        f"\nResolved golden customers: "
        f"{len(momentum):,}"
    )

    print(
        f"Unique golden customer IDs: "
        f"{momentum['golden_customer_id'].nunique():,}"
    )

    print(
        f"Duplicate golden customer IDs: "
        f"{momentum['golden_customer_id'].duplicated().sum():,}"
    )

    print(
        f"Comparable history: "
        f"{momentum['comparable_history'].sum():,}"
    )

    print(
        f"Cadence comparable: "
        f"{momentum['cadence_comparable'].sum():,}"
    )

    print("\nMomentum status:")

    print(
        momentum["momentum_status"]
        .value_counts(dropna=False)
    )

    print("\nMedian change among comparable customers:")

    comparable = momentum.loc[
        momentum["comparable_history"]
    ]

    metrics = [
        "order_change_pct",
        "sales_change_pct",
        "margin_change_pct",
        "aov_change_pct",
        "cadence_change_pct",
    ]

    print(
        comparable[metrics]
        .median()
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading golden customer data...")

    customers = pd.read_parquet(
        CUSTOMER_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    customer_features = pd.read_parquet(
        FEATURE_FILE
    )

    required_customer_columns = {
        "golden_customer_id",
    }

    required_transaction_columns = {
        "golden_customer_id",
        "transaction_date",
        "order_id",
        "net_sales",
        "gross_margin",
        "units",
    }

    required_feature_columns = {
        "golden_customer_id",
        "active_customer",
        "lifecycle_status",
        "days_since_last_purchase",
        "adjusted_lapse_ratio",
        "cadence_confidence",
        "trailing_12m_sales",
        "trailing_12m_margin",
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customers.columns)
    )

    missing_transaction_columns = (
        required_transaction_columns
        - set(transactions.columns)
    )

    missing_feature_columns = (
        required_feature_columns
        - set(customer_features.columns)
    )

    if missing_customer_columns:
        raise ValueError(
            "Golden customer master is missing: "
            f"{sorted(missing_customer_columns)}"
        )

    if missing_transaction_columns:
        raise ValueError(
            "Golden customer transactions are missing: "
            f"{sorted(missing_transaction_columns)}"
        )

    if missing_feature_columns:
        raise ValueError(
            "Customer features are missing: "
            f"{sorted(missing_feature_columns)}"
        )

    if customers["golden_customer_id"].duplicated().any():
        raise ValueError(
            "Golden customer master must contain "
            "one row per golden_customer_id."
        )

    print("Building golden customer momentum features...")

    momentum = build_momentum_features(
        customers,
        transactions,
        customer_features,
    )

    momentum = add_momentum_status(
        momentum
    )

    momentum.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(momentum)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()