from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AS_OF_DATE = pd.Timestamp("2026-07-31")

CUSTOMER_MASTER_FILE = Path(
    "data/runtime/golden_customer_master.parquet"
)

TRANSACTION_FILE = Path(
    "data/runtime/golden_customer_transactions.parquet"
)

FEATURE_FILE = Path(
    "data/runtime/customer_features.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/customer_value.parquet"
)


# ---------------------------------------------------------------------
# RFM base metrics
# ---------------------------------------------------------------------

def build_rfm_base(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    rfm = (
        transactions
        .groupby("golden_customer_id")
        .agg(
            rfm_last_purchase_date=(
                "transaction_date",
                "max",
            ),
            rfm_frequency=(
                "order_id",
                "nunique",
            ),
            rfm_monetary_sales=(
                "net_sales",
                "sum",
            ),
            rfm_monetary_margin=(
                "gross_margin",
                "sum",
            ),
        )
        .reset_index()
    )

    rfm["rfm_recency_days"] = (
        AS_OF_DATE
        - rfm["rfm_last_purchase_date"]
    ).dt.days

    return rfm


# ---------------------------------------------------------------------
# Quantile scoring
# ---------------------------------------------------------------------

def quantile_score(
    series: pd.Series,
    reverse: bool = False,
) -> pd.Series:

    valid = series.dropna()

    if valid.empty:
        return pd.Series(
            np.nan,
            index=series.index,
        )

    ranked = valid.rank(
        method="first"
    )

    scored = pd.qcut(
        ranked,
        q=5,
        labels=[
            1,
            2,
            3,
            4,
            5,
        ],
    ).astype(int)

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float",
    )

    result.loc[valid.index] = scored

    if reverse:
        result.loc[valid.index] = (
            6
            - result.loc[valid.index]
        )

    return result


def add_rfm_scores(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    # Low recency = good, so score is reversed.
    result["r_score"] = quantile_score(
        result["rfm_recency_days"],
        reverse=True,
    )

    result["f_score"] = quantile_score(
        result["rfm_frequency"],
        reverse=False,
    )

    result["m_score"] = quantile_score(
        result["rfm_monetary_margin"],
        reverse=False,
    )

    result["rfm_score"] = (
        result["r_score"].fillna(0)
        + result["f_score"].fillna(0)
        + result["m_score"].fillna(0)
    )

    return result


# ---------------------------------------------------------------------
# Customer value tier
# ---------------------------------------------------------------------

def add_value_tier(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    valid_margin = (
        result["trailing_12m_margin"]
        .fillna(0)
    )

    percentile = valid_margin.rank(
        pct=True,
        method="average",
    )

    conditions = [
        percentile >= 0.90,
        percentile >= 0.70,
        percentile >= 0.40,
    ]

    labels = [
        "Very High",
        "High",
        "Medium",
    ]

    result["customer_value_tier"] = (
        np.select(
            conditions,
            labels,
            default="Low",
        )
    )

    # Never-purchased customers should not be assigned
    # commercial value based on zero trailing margin.
    result.loc[
        result["orders"].fillna(0).eq(0),
        "customer_value_tier",
    ] = "No Purchase"

    return result


# ---------------------------------------------------------------------
# RFM segmentation
# ---------------------------------------------------------------------

def add_rfm_segment(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    no_purchase = result["orders"].fillna(0).eq(0)

    champion = (
        result["r_score"].ge(4)
        & result["f_score"].ge(4)
        & result["m_score"].ge(4)
    )

    loyal = (
        result["r_score"].ge(3)
        & result["f_score"].ge(4)
        & result["m_score"].ge(3)
    )

    high_value = (
        result["m_score"].eq(5)
        & ~champion
    )

    developing = (
        result["r_score"].ge(4)
        & result["f_score"].le(3)
    )

    occasional = (
        result["f_score"].isin(
            [2, 3]
        )
        & result["r_score"].ge(2)
    )

    historically_valuable = (
        result["m_score"].ge(4)
        & result["r_score"].le(2)
    )

    low_engagement = (
        result["f_score"].le(2)
        & result["m_score"].le(2)
    )

    conditions = [
        no_purchase,
        champion,
        loyal,
        historically_valuable,
        high_value,
        developing,
        occasional,
        low_engagement,
    ]

    labels = [
        "No Purchase",
        "Champions",
        "Loyal",
        "Historically Valuable",
        "High Value",
        "Developing",
        "Occasional",
        "Low Engagement",
    ]

    result["rfm_segment"] = np.select(
        conditions,
        labels,
        default="Core",
    )

    return result


# ---------------------------------------------------------------------
# Priority value
# ---------------------------------------------------------------------

def add_commercial_value_score(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    trailing_margin = (
        result["trailing_12m_margin"]
        .fillna(0)
    )

    trailing_sales = (
        result["trailing_12m_sales"]
        .fillna(0)
    )

    result["margin_percentile"] = (
        trailing_margin.rank(
            pct=True,
            method="average",
        )
    )

    result["sales_percentile"] = (
        trailing_sales.rank(
            pct=True,
            method="average",
        )
    )

    result["commercial_value_score"] = (
        0.65
        * result["margin_percentile"]
        + 0.35
        * result["sales_percentile"]
    )

    result["commercial_value_score"] = (
        result["commercial_value_score"]
        * 100
    ).round(1)

    result.loc[
        result["orders"].fillna(0).eq(0),
        "commercial_value_score",
    ] = 0.0

    return result


# ---------------------------------------------------------------------
# Build full customer value table
# ---------------------------------------------------------------------

def build_customer_value(
    customer_master: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> pd.DataFrame:

    rfm = build_rfm_base(
        transactions
    )

    value = customer_master[
        [
            "golden_customer_id",
            "state",
        ]
    ].merge(
        rfm,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    feature_columns = customer_features[
        [
            "golden_customer_id",
            "orders",
            "days_since_last_purchase",
            "active_customer",
            "lifecycle_status",
            "trailing_12m_orders",
            "trailing_12m_sales",
            "trailing_12m_margin",
            "avg_order_value",
        ]
    ]

    value = value.merge(
        feature_columns,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    value = add_rfm_scores(value)

    value = add_value_tier(value)

    value = add_rfm_segment(value)

    value = add_commercial_value_score(
        value
    )

    return value


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    value: pd.DataFrame,
) -> None:

    print("\nCUSTOMER VALUE QA")
    print("=" * 75)

    print(
        f"Resolved golden customers: "
        f"{len(value):,}"
    )

    print(
        f"Unique golden customer IDs: "
        f"{value['golden_customer_id'].nunique():,}"
    )

    print(
        f"Duplicate golden customer IDs: "
        f"{value['golden_customer_id'].duplicated().sum():,}"
    )

    print(
        f"Never-purchased golden customers: "
        f"{value['orders'].fillna(0).eq(0).sum():,}"
    )

    print("\nRFM segment:")

    print(
        value["rfm_segment"]
        .value_counts(dropna=False)
    )

    print("\nCustomer value tier:")

    print(
        value["customer_value_tier"]
        .value_counts(dropna=False)
    )

    print(
        "\nMedian trailing 12M margin "
        "by value tier:"
    )

    print(
        value
        .groupby(
            "customer_value_tier"
        )["trailing_12m_margin"]
        .median()
        .round(2)
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nMedian commercial value score "
        "by RFM segment:"
    )

    print(
        value
        .groupby(
            "rfm_segment"
        )["commercial_value_score"]
        .median()
        .round(1)
        .sort_values(
            ascending=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading golden customer data...")

    customer_master = pd.read_parquet(
        CUSTOMER_MASTER_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    customer_features = pd.read_parquet(
        FEATURE_FILE
    )

    required_master_columns = {
        "golden_customer_id",
        "state",
    }

    required_transaction_columns = {
        "golden_customer_id",
        "transaction_date",
        "order_id",
        "net_sales",
        "gross_margin",
    }

    required_feature_columns = {
        "golden_customer_id",
        "orders",
        "days_since_last_purchase",
        "active_customer",
        "lifecycle_status",
        "trailing_12m_orders",
        "trailing_12m_sales",
        "trailing_12m_margin",
        "avg_order_value",
    }

    missing_master = (
        required_master_columns
        - set(customer_master.columns)
    )

    missing_transactions = (
        required_transaction_columns
        - set(transactions.columns)
    )

    missing_features = (
        required_feature_columns
        - set(customer_features.columns)
    )

    if missing_master:
        raise ValueError(
            "Golden customer master is missing: "
            f"{sorted(missing_master)}"
        )

    if missing_transactions:
        raise ValueError(
            "Golden customer transactions are missing: "
            f"{sorted(missing_transactions)}"
        )

    if missing_features:
        raise ValueError(
            "Customer features are missing: "
            f"{sorted(missing_features)}"
        )

    if customer_master["golden_customer_id"].duplicated().any():
        raise ValueError(
            "Golden customer master must contain "
            "one row per golden_customer_id."
        )

    print("Building golden customer value features...")

    value = build_customer_value(
        customer_master,
        transactions,
        customer_features,
    )

    value.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(value)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()