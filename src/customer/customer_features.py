from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AS_OF_DATE = pd.Timestamp("2026-07-31")

INPUT_DIR = Path("data/runtime")
OUTPUT_DIR = Path("data/runtime")

CUSTOMER_FILE = INPUT_DIR / "golden_customer_master.parquet"
TRANSACTION_FILE = INPUT_DIR / "golden_customer_transactions.parquet"

OUTPUT_FILE = OUTPUT_DIR / "customer_features.parquet"


# ---------------------------------------------------------------------
# Core customer metrics
# ---------------------------------------------------------------------

def build_customer_summary(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    summary = (
        transactions
        .groupby("golden_customer_id")
        .agg(
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
            orders=("order_id", "nunique"),
            units=("units", "sum"),
            net_sales=("net_sales", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    summary["avg_order_value"] = (
        summary["net_sales"] / summary["orders"]
    )

    summary["days_since_last_purchase"] = (
        AS_OF_DATE - summary["last_purchase_date"]
    ).dt.days

    summary["customer_tenure_days"] = (
        AS_OF_DATE - summary["first_purchase_date"]
    ).dt.days

    result = customers[
        [
            "golden_customer_id",
            "state",
        ]
    ].merge(
        summary,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    # Golden identities with no observed purchases remain part of
    # the analytical customer universe. Monetary/order measures are
    # zero; purchase dates and recency remain null.
    zero_fill_columns = [
        "orders",
        "units",
        "net_sales",
        "gross_margin",
        "avg_order_value",
    ]

    for column in zero_fill_columns:
        result[column] = (
            result[column]
            .fillna(0)
        )

    return result


# ---------------------------------------------------------------------
# Purchase cadence
# ---------------------------------------------------------------------

def build_purchase_gaps(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    purchase_dates = (
        transactions[
            ["golden_customer_id", "transaction_date"]
        ]
        .drop_duplicates()
        .sort_values(
            ["golden_customer_id", "transaction_date"]
        )
    )

    purchase_dates["previous_purchase_date"] = (
        purchase_dates
        .groupby("golden_customer_id")["transaction_date"]
        .shift(1)
    )

    purchase_dates["purchase_gap_days"] = (
        purchase_dates["transaction_date"]
        - purchase_dates["previous_purchase_date"]
    ).dt.days

    return purchase_dates.dropna(
        subset=["purchase_gap_days"]
    )


def build_cadence_features(
    purchase_gaps: pd.DataFrame,
) -> pd.DataFrame:

    cadence = (
        purchase_gaps
        .groupby("golden_customer_id")
        .agg(
            observed_purchase_gaps=(
                "purchase_gap_days",
                "count",
            ),
            median_purchase_gap_days=(
                "purchase_gap_days",
                "median",
            ),
            mean_purchase_gap_days=(
                "purchase_gap_days",
                "mean",
            ),
            purchase_gap_std_days=(
                "purchase_gap_days",
                "std",
            ),
        )
        .reset_index()
    )

    cadence["cadence_cv"] = (
        cadence["purchase_gap_std_days"]
        / cadence["mean_purchase_gap_days"]
    )

    return cadence


# ---------------------------------------------------------------------
# Recent customer value
# ---------------------------------------------------------------------

def build_trailing_value(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    trailing_start = (
        AS_OF_DATE - pd.DateOffset(months=12)
    )

    trailing = transactions.loc[
        transactions["transaction_date"] > trailing_start
    ]

    value = (
        trailing
        .groupby("golden_customer_id")
        .agg(
            trailing_12m_orders=("order_id", "nunique"),
            trailing_12m_sales=("net_sales", "sum"),
            trailing_12m_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    return value


# ---------------------------------------------------------------------
# Lapse measures
# ---------------------------------------------------------------------

def add_lapse_features(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    result["lapse_ratio"] = (
        result["days_since_last_purchase"]
        / result["median_purchase_gap_days"]
    )

    # Guard against extremely short cadence creating hypersensitive
    # lapse signals.
    result["expected_return_days"] = (
        result["median_purchase_gap_days"]
        .clip(lower=7)
    )

    result["adjusted_lapse_ratio"] = (
        result["days_since_last_purchase"]
        / result["expected_return_days"]
    )

    return result


# ---------------------------------------------------------------------
# Evidence / confidence
# ---------------------------------------------------------------------

def add_cadence_confidence(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    conditions = [
        result["observed_purchase_gaps"].fillna(0) >= 5,
        result["observed_purchase_gaps"].fillna(0) >= 2,
    ]

    labels = [
        "High",
        "Medium",
    ]

    result["cadence_confidence"] = np.select(
        conditions,
        labels,
        default="Low",
    )

    return result


# ---------------------------------------------------------------------
# Business-defined active customer
# ---------------------------------------------------------------------

def add_active_flag(
    customers: pd.DataFrame,
    active_window_days: int = 365,
) -> pd.DataFrame:

    result = customers.copy()

    result["active_customer"] = (
        result["days_since_last_purchase"]
        <= active_window_days
    )

    result["active_window_days"] = active_window_days

    return result


# ---------------------------------------------------------------------
# Lifecycle / lapse status
# ---------------------------------------------------------------------

def add_lifecycle_status(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    no_purchase = result["orders"].fillna(0).eq(0)

    low_evidence = (
        result["cadence_confidence"].eq("Low")
    )

    ratio = result["adjusted_lapse_ratio"]

    conditions = [
        no_purchase,
        low_evidence & result["active_customer"],
        low_evidence & ~result["active_customer"],
        ratio < 1.0,
        ratio < 1.5,
        ratio < 2.0,
        ratio >= 2.0,
    ]

    labels = [
        "Acquired / Never Purchased",
        "Active / Limited History",
        "Inactive / Limited History",
        "On Cadence",
        "Watch",
        "At Risk",
        "Highly Lapsed",
    ]

    result["lifecycle_status"] = np.select(
        conditions,
        labels,
        default="Unclassified",
    )

    return result


# ---------------------------------------------------------------------
# Build complete feature table
# ---------------------------------------------------------------------

def build_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    customer_summary = build_customer_summary(
        customers,
        transactions,
    )

    purchase_gaps = build_purchase_gaps(
        transactions
    )

    cadence = build_cadence_features(
        purchase_gaps
    )

    trailing_value = build_trailing_value(
        transactions
    )

    features = customer_summary.merge(
        cadence,
        on="golden_customer_id",
        how="left",
    )

    features = features.merge(
        trailing_value,
        on="golden_customer_id",
        how="left",
    )

    features = add_lapse_features(features)

    features = add_cadence_confidence(features)

    features = add_active_flag(
        features,
        active_window_days=365,
    )

    features = add_lifecycle_status(features)

    return features


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    features: pd.DataFrame,
) -> None:

    print("\nCUSTOMER FEATURE QA")
    print("-" * 60)

    print(
        f"Resolved golden customers: "
        f"{len(features):,}"
    )

    print(
        f"Golden customers with purchases: "
        f"{features['orders'].gt(0).sum():,}"
    )

    print(
        f"Golden customers never purchased: "
        f"{features['orders'].eq(0).sum():,}"
    )

    print(
        f"Unique golden customer IDs: "
        f"{features['golden_customer_id'].nunique():,}"
    )

    print(
        f"Active customers: "
        f"{features['active_customer'].sum():,}"
    )

    print(
        f"Customers with cadence estimate: "
        f"{features['median_purchase_gap_days'].notna().sum():,}"
    )

    print("\nCadence confidence:")
    print(
        features["cadence_confidence"]
        .value_counts(dropna=False)
    )

    print("\nLifecycle status:")
    print(
        features["lifecycle_status"]
        .value_counts(dropna=False)
    )

    duplicate_ids = (
        features["golden_customer_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate golden customer IDs: "
        f"{duplicate_ids:,}"
    )

    print("\nMedian metrics:")

    print(
        features[
            [
                "days_since_last_purchase",
                "median_purchase_gap_days",
                "adjusted_lapse_ratio",
                "trailing_12m_sales",
            ]
        ]
        .median(numeric_only=True)
        .round(2)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
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

    required_customer_columns = {
        "golden_customer_id",
        "state",
    }

    required_transaction_columns = {
        "golden_customer_id",
        "order_id",
        "transaction_date",
        "units",
        "net_sales",
        "gross_margin",
    }

    missing_customer_columns = (
        required_customer_columns
        - set(customers.columns)
    )

    missing_transaction_columns = (
        required_transaction_columns
        - set(transactions.columns)
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

    if customers["golden_customer_id"].duplicated().any():
        raise ValueError(
            "Golden customer master must contain "
            "one row per golden_customer_id."
        )

    print("Building golden customer features...")

    features = build_customer_features(
        customers,
        transactions,
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