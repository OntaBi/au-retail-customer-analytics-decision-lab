from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRANSACTION_FILE = Path("data/runtime/golden_customer_transactions.parquet")
EXPOSURE_FILE = Path("data/generated/campaign_exposures.parquet")
OUTPUT_FILE = Path("data/runtime/promotion_response_training.parquet")

LOOKBACK_DAYS = 180
MIN_HISTORY_DAYS = 90


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = pd.read_parquet(TRANSACTION_FILE).copy()
    exp = pd.read_parquet(EXPOSURE_FILE).copy()

    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    exp["campaign_date"] = pd.to_datetime(exp["campaign_date"])

    required_tx = {
        "golden_customer_id", "transaction_date", "order_id",
        "net_sales", "gross_margin", "discount_pct", "channel", "category",
    }
    required_exp = {
        "exposure_id", "campaign_id", "campaign_date", "golden_customer_id",
        "treatment_flag", "campaign_channel", "offer_category",
        "discount_depth", "responded",
    }

    missing_tx = required_tx - set(tx.columns)
    missing_exp = required_exp - set(exp.columns)

    if missing_tx:
        raise ValueError(f"Transactions missing columns: {sorted(missing_tx)}")
    if missing_exp:
        raise ValueError(f"Campaign exposures missing columns: {sorted(missing_exp)}")

    tx["discounted_order"] = tx["discount_pct"].fillna(0).gt(0)
    return tx, exp


def build_snapshot_features(
    tx: pd.DataFrame,
    campaign_date: pd.Timestamp,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    history = tx.loc[tx["transaction_date"].lt(campaign_date)].copy()
    if history.empty:
        return pd.DataFrame()

    lifetime = (
        history.groupby("golden_customer_id")
        .agg(
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
            lifetime_orders=("order_id", "nunique"),
            lifetime_sales=("net_sales", "sum"),
            lifetime_margin=("gross_margin", "sum"),
            average_order_value=("net_sales", "mean"),
            average_discount_pct=("discount_pct", "mean"),
            discounted_order_share=("discounted_order", "mean"),
            lifetime_categories=("category", "nunique"),
            lifetime_channels=("channel", "nunique"),
        )
        .reset_index()
    )

    lifetime["days_since_last_purchase"] = (
        campaign_date - lifetime["last_purchase_date"]
    ).dt.days
    lifetime["customer_tenure_days"] = (
        campaign_date - lifetime["first_purchase_date"]
    ).dt.days
    lifetime["lifetime_margin_rate"] = safe_divide(
        lifetime["lifetime_margin"], lifetime["lifetime_sales"]
    )

    lookback_start = campaign_date - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = history.loc[
        history["transaction_date"].ge(lookback_start)
    ].copy()

    recent_agg = (
        recent.groupby("golden_customer_id")
        .agg(
            orders_prior_180d=("order_id", "nunique"),
            sales_prior_180d=("net_sales", "sum"),
            margin_prior_180d=("gross_margin", "sum"),
            categories_prior_180d=("category", "nunique"),
            channels_prior_180d=("channel", "nunique"),
            avg_discount_prior_180d=("discount_pct", "mean"),
            discounted_order_share_prior_180d=("discounted_order", "mean"),
        )
        .reset_index()
    )

    # Preferred historical channel.
    channel_counts = (
        history.groupby(["golden_customer_id", "channel"])
        .agg(channel_orders=("order_id", "nunique"))
        .reset_index()
        .sort_values(
            ["golden_customer_id", "channel_orders", "channel"],
            ascending=[True, False, True],
        )
        .drop_duplicates("golden_customer_id")
        [["golden_customer_id", "channel"]]
        .rename(columns={"channel": "preferred_channel"})
    )

    # Historical affinity to each offer category.
    category_counts = (
        history.groupby(["golden_customer_id", "category"])
        .agg(
            category_orders=("order_id", "nunique"),
            category_sales=("net_sales", "sum"),
        )
        .reset_index()
    )

    total_orders = (
        history.groupby("golden_customer_id")["order_id"]
        .nunique()
        .rename("all_orders")
        .reset_index()
    )
    category_counts = category_counts.merge(
        total_orders, on="golden_customer_id", how="left"
    )
    category_counts["category_order_share"] = safe_divide(
        category_counts["category_orders"], category_counts["all_orders"]
    )

    frame = customers.merge(
        lifetime, on="golden_customer_id", how="left", validate="many_to_one"
    )
    frame = frame.merge(
        recent_agg, on="golden_customer_id", how="left", validate="many_to_one"
    )
    frame = frame.merge(
        channel_counts, on="golden_customer_id", how="left", validate="many_to_one"
    )

    frame = frame.merge(
        category_counts[
            ["golden_customer_id", "category", "category_orders",
             "category_sales", "category_order_share"]
        ],
        left_on=["golden_customer_id", "offer_category"],
        right_on=["golden_customer_id", "category"],
        how="left",
        validate="many_to_one",
    )

    frame["offer_category_prior_orders"] = frame["category_orders"].fillna(0)
    frame["offer_category_prior_sales"] = frame["category_sales"].fillna(0)
    frame["offer_category_order_share"] = frame["category_order_share"].fillna(0)
    frame["offer_category_seen_before"] = (
        frame["offer_category_prior_orders"].gt(0).astype(int)
    )
    frame["campaign_channel_matches_preference"] = (
        frame["campaign_channel"].eq(frame["preferred_channel"]).astype(int)
    )

    for col in [
        "orders_prior_180d", "sales_prior_180d", "margin_prior_180d",
        "categories_prior_180d", "channels_prior_180d",
        "avg_discount_prior_180d", "discounted_order_share_prior_180d",
    ]:
        frame[col] = frame[col].fillna(0)

    frame["recent_margin_rate"] = safe_divide(
        frame["margin_prior_180d"], frame["sales_prior_180d"]
    ).fillna(0)

    frame["orders_per_30d_tenure"] = safe_divide(
        frame["lifetime_orders"],
        frame["customer_tenure_days"].clip(lower=1) / 30.0,
    ).fillna(0)

    frame["days_since_last_purchase"] = frame[
        "days_since_last_purchase"
    ].fillna(9999)

    return frame


def build_training_data(
    tx: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for campaign_date in sorted(exposures["campaign_date"].unique()):
        campaign_date = pd.Timestamp(campaign_date)
        cohort = exposures.loc[
            exposures["campaign_date"].eq(campaign_date)
        ].copy()

        snapshot = build_snapshot_features(tx, campaign_date, cohort)
        if snapshot.empty:
            continue

        snapshot = snapshot.loc[
            snapshot["customer_tenure_days"].ge(MIN_HISTORY_DAYS)
            & snapshot["lifetime_orders"].ge(2)
        ].copy()

        frames.append(snapshot)

    if not frames:
        raise ValueError("No promotion-response training rows were produced.")

    training = pd.concat(frames, ignore_index=True)

    drop_helpers = [
        "category", "category_orders", "category_sales",
        "category_order_share", "first_purchase_date", "last_purchase_date",
        "response_window_end",
    ]
    training = training.drop(
        columns=[c for c in drop_helpers if c in training.columns]
    )

    training = training.sort_values(
        ["campaign_date", "exposure_id"]
    ).reset_index(drop=True)

    if training["exposure_id"].duplicated().any():
        raise ValueError("Duplicate exposure_id found in training data.")

    return training


def run_qa(training: pd.DataFrame) -> None:
    print()
    print("=" * 80)
    print("PROMOTION RESPONSE PROPENSITY TRAINING QA")
    print("=" * 80)

    print(f"Training rows                 : {len(training):,}")
    print(
        f"Unique golden customers       : "
        f"{training['golden_customer_id'].nunique():,}"
    )
    print(
        f"Campaigns                     : "
        f"{training['campaign_id'].nunique():,}"
    )
    print(
        f"Observation range             : "
        f"{training['campaign_date'].min().date()} to "
        f"{training['campaign_date'].max().date()}"
    )
    print(f"Overall response rate         : {training['responded'].mean():.1%}")

    treated = training.loc[training["treatment_flag"].eq(1)]
    control = training.loc[training["treatment_flag"].eq(0)]

    print(f"Treated response rate         : {treated['responded'].mean():.1%}")
    print(f"Control response rate         : {control['responded'].mean():.1%}")
    print(
        f"Observed treatment lift       : "
        f"{treated['responded'].mean() - control['responded'].mean():+.1%}"
    )
    print(
        f"Median days since purchase    : "
        f"{training['days_since_last_purchase'].median():.0f}"
    )
    print(
        f"Median prior 180d orders      : "
        f"{training['orders_prior_180d'].median():.0f}"
    )
    print(
        f"Offer category seen before    : "
        f"{training['offer_category_seen_before'].mean():.1%}"
    )
    print(
        f"Channel preference match      : "
        f"{training['campaign_channel_matches_preference'].mean():.1%}"
    )

    print()
    print("Rows by campaign:")
    summary = (
        training.groupby(["campaign_date", "treatment_flag"])
        .agg(
            rows=("exposure_id", "size"),
            customers=("golden_customer_id", "nunique"),
            positive_rate=("responded", "mean"),
            median_days_since=("days_since_last_purchase", "median"),
            median_prior_orders=("orders_prior_180d", "median"),
        )
        .reset_index()
    )
    summary["positive_rate"] = summary["positive_rate"].map(lambda x: f"{x:.1%}")
    print(summary.to_string(index=False))

    prohibited = {
        "response_probability_used",
        "incremental_probability_truth",
        "natural_purchase_in_window",
        "natural_category_purchase_in_window",
    }
    leaked = sorted(prohibited.intersection(training.columns))

    if leaked:
        raise ValueError(
            f"Leakage guard failed. Hidden truth fields present: {leaked}"
        )

    outcome_fields = ["responded"]
    print()
    print(
        "Leakage guard: hidden synthetic truth absent. "
        f"Outcome field retained only as target: {outcome_fields}"
    )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading campaign exposures and golden customer transactions...")
    tx, exposures = load_inputs()

    print("Reconstructing pre-campaign customer features...")
    training = build_training_data(tx, exposures)

    training.to_parquet(OUTPUT_FILE, index=False)

    run_qa(training)

    print()
    print("File created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
