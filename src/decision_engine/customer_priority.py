from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VALUE_FILE = Path(
    "data/runtime/customer_value.parquet"
)

FEATURE_FILE = Path(
    "data/runtime/customer_features.parquet"
)

MOMENTUM_FILE = Path(
    "data/runtime/customer_momentum.parquet"
)

CADENCE_TREND_FILE = Path(
    "data/runtime/customer_cadence_trend.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/customer_priority.parquet"
)


# ---------------------------------------------------------------------
# Lapse risk score
# ---------------------------------------------------------------------

def build_lapse_score(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    ratio = (
        result["adjusted_lapse_ratio"]
        .fillna(0)
    )

    # Convert behavioural lapse ratio into a transparent 0-100 score.
    #
    # 0.0x expected cadence -> 0
    # 1.0x expected cadence -> 40
    # 1.5x expected cadence -> 60
    # 2.0x expected cadence -> 80
    # 3.0x+ expected cadence -> 100

    result["lapse_risk_score"] = np.select(
        [
            ratio < 1.0,
            ratio < 1.5,
            ratio < 2.0,
            ratio < 3.0,
        ],
        [
            ratio * 40,
            40 + ((ratio - 1.0) / 0.5) * 20,
            60 + ((ratio - 1.5) / 0.5) * 20,
            80 + (ratio - 2.0) * 20,
        ],
        default=100,
    )

    result["lapse_risk_score"] = (
        result["lapse_risk_score"]
        .clip(0, 100)
        .round(1)
    )

    # Do not assign behavioural lapse risk when there is
    # insufficient purchase history.
    low_confidence = (
        result["cadence_confidence"]
        .eq("Low")
    )

    result.loc[
        low_confidence,
        "lapse_risk_score",
    ] = 0.0

    return result


# ---------------------------------------------------------------------
# Momentum risk score
# ---------------------------------------------------------------------

def build_momentum_score(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    status_score = {
        "Strongly Deteriorating": 100,
        "Deteriorating": 70,
        "Stable / Mixed": 20,
        "Improving": 0,
        "Insufficient History": 0,
    }

    result["momentum_risk_score"] = (
        result["cadence_trend_status"]
        .map(status_score)
        .fillna(0)
        .astype(float)
    )

    return result


# ---------------------------------------------------------------------
# Commercial value score
# ---------------------------------------------------------------------

def build_value_score(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    result["value_score"] = (
        result["commercial_value_score"]
        .fillna(0)
        .clip(0, 100)
    )

    return result


# ---------------------------------------------------------------------
# Composite priority score
# ---------------------------------------------------------------------

def build_priority_score(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    result["priority_score"] = (
        0.40 * result["value_score"]
        + 0.35 * result["lapse_risk_score"]
        + 0.25 * result["momentum_risk_score"]
    )

    result["priority_score"] = (
        result["priority_score"]
        .clip(0, 100)
        .round(1)
    )

    return result


# ---------------------------------------------------------------------
# Decision groups
# ---------------------------------------------------------------------

def add_decision_group(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    high_value = (
        result["customer_value_tier"]
        .isin(
            [
                "High",
                "Very High",
            ]
        )
    )

    meaningful_value = (
        result["customer_value_tier"]
        .isin(
            [
                "Medium",
                "High",
                "Very High",
            ]
        )
    )

    hard_lapse = (
        result["lifecycle_status"]
        .isin(
            [
                "At Risk",
                "Highly Lapsed",
            ]
        )
    )

    early_lapse = (
        result["lifecycle_status"]
        .eq("Watch")
    )

    deteriorating = (
        result["cadence_trend_status"]
        .isin(
            [
                "Deteriorating",
                "Strongly Deteriorating",
            ]
        )
    )

    strong_deterioration = (
        result["cadence_trend_status"]
        .eq("Strongly Deteriorating")
    )

    healthy_lapse = (
        result["lifecycle_status"]
        .eq("On Cadence")
    )

    improving = (
        result["cadence_trend_status"]
        .eq("Improving")
    )

    no_purchase = (
        result["rfm_segment"]
        .eq("No Purchase")
    )

    developing = (
        result["rfm_segment"]
        .eq("Developing")
    )

    conditions = [
        no_purchase,

        high_value & hard_lapse,

        high_value
        & healthy_lapse
        & deteriorating,

        meaningful_value
        & strong_deterioration,

        meaningful_value
        & hard_lapse,

        meaningful_value
        & (
            early_lapse
            | deteriorating
        ),

        developing
        & result["active_customer"],

        high_value
        & healthy_lapse
        & ~deteriorating,

        improving
        & meaningful_value,
    ]

    labels = [
        "Acquisition Opportunity",
        "Protect Now",
        "Proactive Retention",
        "Proactive Retention",
        "Re-engage",
        "Watch Closely",
        "Develop",
        "Maintain",
        "Maintain",
    ]

    result["decision_group"] = np.select(
        conditions,
        labels,
        default="Low Priority",
    )

    return result


# ---------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------

def add_recommended_action(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    action_map = {
        "Protect Now":
            "Prioritise immediate personalised retention action",

        "Proactive Retention":
            "Intervene before lapse; investigate declining engagement",

        "Re-engage":
            "Target with relevant re-engagement offer or communication",

        "Watch Closely":
            "Monitor behaviour and trigger action if deterioration continues",

        "Develop":
            "Encourage second or next purchase and deepen engagement",

        "Maintain":
            "Maintain relationship and protect current customer value",

        "Acquisition Opportunity":
            "Encourage first purchase and assess acquisition effectiveness",

        "Low Priority":
            "No immediate intervention; retain in standard lifecycle activity",
    }

    result["recommended_action"] = (
        result["decision_group"]
        .map(action_map)
    )

    return result


# ---------------------------------------------------------------------
# Priority band
# ---------------------------------------------------------------------

def add_priority_band(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    result = customers.copy()

    conditions = [
        result["priority_score"] >= 75,
        result["priority_score"] >= 55,
        result["priority_score"] >= 35,
    ]

    labels = [
        "Critical",
        "High",
        "Medium",
    ]

    result["priority_band"] = np.select(
        conditions,
        labels,
        default="Low",
    )

    return result


# ---------------------------------------------------------------------
# Build decision table
# ---------------------------------------------------------------------

def build_customer_priority(
    value: pd.DataFrame,
    features: pd.DataFrame,
    momentum: pd.DataFrame,
    cadence_trend: pd.DataFrame,
) -> pd.DataFrame:

    # Customer value table is the primary customer-level foundation.
    result = value.copy()

    feature_columns = features[
        [
            "customer_id",
            "median_purchase_gap_days",
            "expected_return_days",
            "adjusted_lapse_ratio",
            "cadence_confidence",
        ]
    ]

    result = result.merge(
        feature_columns,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Keep the descriptive 180-day period comparison.
    momentum_columns = momentum[
        [
            "customer_id",
            "order_change_pct",
            "sales_change_pct",
            "margin_change_pct",
            "aov_change_pct",
            "cadence_change_pct",
            "momentum_status",
        ]
    ]

    result = result.merge(
        momentum_columns,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    # Analytical cadence trend signal.
    trend_columns = cadence_trend[
        [
            "customer_id",
            "trend_gap_count",
            "gap_trend_slope",
            "gap_trend_pct",
            "gap_trend_r_squared",
            "recent_vs_early_gap_pct",
            "cadence_trend_status",
        ]
    ]

    result = result.merge(
        trend_columns,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    result = build_value_score(result)
    result = build_lapse_score(result)
    result = build_momentum_score(result)
    result = build_priority_score(result)

    result = add_decision_group(result)
    result = add_recommended_action(result)
    result = add_priority_band(result)

    return result


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    priority: pd.DataFrame,
) -> None:

    print("\nCUSTOMER PRIORITY QA")
    print("=" * 80)

    print(
        f"Customers: "
        f"{len(priority):,}"
    )

    print("\nDecision groups:")

    print(
        priority["decision_group"]
        .value_counts(dropna=False)
    )

    print("\nPriority bands:")

    print(
        priority["priority_band"]
        .value_counts(dropna=False)
    )

    print(
        "\nMedian priority score "
        "by decision group:"
    )

    print(
        priority
        .groupby(
            "decision_group"
        )["priority_score"]
        .median()
        .round(1)
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nTrailing 12M margin "
        "by decision group:"
    )

    margin_summary = (
        priority
        .groupby("decision_group")
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),
            trailing_12m_margin=(
                "trailing_12m_margin",
                "sum",
            ),
        )
        .sort_values(
            "trailing_12m_margin",
            ascending=False,
        )
    )

    margin_summary[
        "trailing_12m_margin"
    ] = (
        margin_summary[
            "trailing_12m_margin"
        ]
        .round(0)
    )

    print(margin_summary)

    print(
        "\nTop 10 priority customers:"
    )

    display_columns = [
        "customer_id",
        "rfm_segment",
        "customer_value_tier",
        "lifecycle_status",
        "cadence_trend_status",
        "decision_group",
        "priority_score",
        "trailing_12m_sales",
        "trailing_12m_margin",
    ]

    print(
        priority[
            display_columns
        ]
        .sort_values(
            "priority_score",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading customer analytics...")

    value = pd.read_parquet(
        VALUE_FILE
    )

    features = pd.read_parquet(
        FEATURE_FILE
    )

    momentum = pd.read_parquet(
        MOMENTUM_FILE
    )

    cadence_trend = pd.read_parquet(
        CADENCE_TREND_FILE
    )

    print("Building customer priority decisions...")

    priority = build_customer_priority(
        value,
        features,
        momentum,
        cadence_trend,
    )

    priority.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(priority)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()