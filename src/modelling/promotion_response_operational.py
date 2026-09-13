from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.modelling.promotion_response_training import build_snapshot_features
from src.modelling.promotion_response_propensity import (
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    add_time_features,
    build_logistic_model,
    build_gradient_boosting_model,
    fit_sigmoid_calibrator,
    calibrate_probability,
    load_training_data,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRANSACTION_FILE = Path(
    "data/runtime/golden_customer_transactions.parquet"
)

PRIORITY_FILE = Path(
    "data/runtime/customer_priority.parquet"
)

GOVERNANCE_FILE = Path(
    "data/runtime/promotion_response_model_governance.parquet"
)

METADATA_FILE = Path(
    "data/runtime/promotion_response_model_metadata.json"
)

OUTPUT_FILE = Path(
    "data/runtime/promotion_response_operational_scores.parquet"
)

AS_OF_DATE = pd.Timestamp("2026-07-31")
LATEST_CALIBRATION_DATE = pd.Timestamp("2026-05-15")

CAMPAIGN_CHANNELS = [
    "Email",
    "SMS",
    "App",
]

DISCOUNT_DEPTHS = [
    0.10,
    0.15,
    0.20,
    0.25,
]

ELIGIBLE_VALUE_TIERS = {
    "Medium",
    "High",
    "Very High",
}


# ---------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------

def model_is_accepted() -> bool:
    if not GOVERNANCE_FILE.exists():
        return False

    governance = pd.read_parquet(
        GOVERNANCE_FILE
    )

    if governance.empty or "pass" not in governance.columns:
        return False

    passed = governance["pass"]

    if passed.dtype != bool:
        passed = (
            passed.astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    return bool(passed.all())


def selected_candidate() -> str:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing model metadata: {METADATA_FILE}"
        )

    metadata = json.loads(
        METADATA_FILE.read_text(
            encoding="utf-8"
        )
    )

    return str(
        metadata[
            "selected_candidate"
        ]
    )


# ---------------------------------------------------------------------
# Current customer / offer candidates
# ---------------------------------------------------------------------

def build_dominant_category(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    tx = transactions.loc[
        pd.to_datetime(
            transactions["transaction_date"]
        ).le(AS_OF_DATE)
    ].copy()

    profile = (
        tx.groupby(
            [
                "golden_customer_id",
                "category",
            ]
        )
        .agg(
            category_orders=(
                "order_id",
                "nunique",
            ),
            category_margin=(
                "gross_margin",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "golden_customer_id",
                "category_orders",
                "category_margin",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .drop_duplicates(
            "golden_customer_id"
        )
        [
            [
                "golden_customer_id",
                "category",
            ]
        ]
        .rename(
            columns={
                "category":
                    "offer_category"
            }
        )
    )

    return profile


def build_offer_candidates(
    priority: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    eligible = priority.loc[
        priority[
            "active_customer"
        ]
        .fillna(False)
        & priority[
            "customer_value_tier"
        ]
        .isin(
            ELIGIBLE_VALUE_TIERS
        )
    ][
        [
            "golden_customer_id",
        ]
    ].drop_duplicates()

    dominant = build_dominant_category(
        transactions
    )

    eligible = eligible.merge(
        dominant,
        on="golden_customer_id",
        how="inner",
        validate="one_to_one",
    )

    channel_frame = pd.DataFrame(
        {
            "campaign_channel":
                CAMPAIGN_CHANNELS
        }
    )

    discount_frame = pd.DataFrame(
        {
            "discount_depth":
                DISCOUNT_DEPTHS
        }
    )

    candidates = (
        eligible
        .merge(
            channel_frame,
            how="cross",
        )
        .merge(
            discount_frame,
            how="cross",
        )
    )

    candidates[
        "campaign_date"
    ] = AS_OF_DATE

    candidates[
        "campaign_id"
    ] = "OPERATIONAL_20260731"

    return candidates


# ---------------------------------------------------------------------
# Final operational model fit
# ---------------------------------------------------------------------

def fit_operational_model():

    if not model_is_accepted():
        raise ValueError(
            "Promotion response model is not ACCEPTED by governance. "
            "Operational scoring is blocked."
        )

    data = load_training_data()

    train = data.loc[
        data[
            "campaign_date"
        ]
        .lt(
            LATEST_CALIBRATION_DATE
        )
    ].copy()

    calibration = data.loc[
        data[
            "campaign_date"
        ]
        .eq(
            LATEST_CALIBRATION_DATE
        )
    ].copy()

    if train.empty or calibration.empty:
        raise ValueError(
            "Operational promotion model requires historical training "
            "data plus the latest calibration campaign."
        )

    candidate_name = selected_candidate()

    if candidate_name == "LogisticRegression":
        model = build_logistic_model()
    elif candidate_name == "HistGradientBoosting":
        model = build_gradient_boosting_model()
    else:
        raise ValueError(
            f"Unsupported selected candidate: {candidate_name}"
        )

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    model.fit(
        train[
            feature_columns
        ],
        train[
            TARGET
        ],
    )

    calibration_raw = (
        model.predict_proba(
            calibration[
                feature_columns
            ]
        )[:, 1]
    )

    calibrator = fit_sigmoid_calibrator(
        calibration_raw,
        calibration[
            TARGET
        ],
    )

    return (
        model,
        calibrator,
        candidate_name,
    )


# ---------------------------------------------------------------------
# Current scoring
# ---------------------------------------------------------------------

def build_operational_scores(
    transactions: pd.DataFrame,
    priority: pd.DataFrame,
) -> pd.DataFrame:

    (
        model,
        calibrator,
        candidate_name,
    ) = fit_operational_model()

    offer_candidates = (
        build_offer_candidates(
            priority,
            transactions,
        )
    )

    snapshot = build_snapshot_features(
        transactions,
        AS_OF_DATE,
        offer_candidates,
    )

    snapshot[
        "campaign_date"
    ] = AS_OF_DATE

    snapshot = add_time_features(
        snapshot
    )

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    raw_probability = (
        model.predict_proba(
            snapshot[
                feature_columns
            ]
        )[:, 1]
    )

    calibrated_probability = (
        calibrate_probability(
            calibrator,
            raw_probability,
        )
    )

    snapshot[
        "promotion_response_propensity_raw"
    ] = raw_probability

    snapshot[
        "promotion_response_propensity"
    ] = calibrated_probability

    margin_after_discount = (
        snapshot[
            "lifetime_margin_rate"
        ]
        .fillna(
            snapshot[
                "recent_margin_rate"
            ]
        )
        .fillna(0.30)
        - snapshot[
            "discount_depth"
        ]
    ).clip(
        lower=0.01
    )

    snapshot[
        "offer_scenario_expected_contribution"
    ] = (
        snapshot[
            "promotion_response_propensity"
        ]
        * snapshot[
            "average_order_value"
        ]
        .fillna(0)
        * margin_after_discount
    )

    ranked = (
        snapshot
        .sort_values(
            [
                "golden_customer_id",
                "offer_scenario_expected_contribution",
                "promotion_response_propensity",
                "discount_depth",
            ],
            ascending=[
                True,
                False,
                False,
                True,
            ],
        )
        .copy()
    )

    best = (
        ranked
        .drop_duplicates(
            "golden_customer_id",
            keep="first",
        )
        .copy()
    )

    output_columns = [
        "golden_customer_id",
        "campaign_date",
        "campaign_channel",
        "offer_category",
        "discount_depth",
        "promotion_response_propensity",
        "promotion_response_propensity_raw",
        "offer_scenario_expected_contribution",
    ]

    best = best[
        output_columns
    ].copy()

    best[
        "model_status"
    ] = "ACCEPTED"

    best[
        "model_candidate"
    ] = candidate_name

    best[
        "eligible_for_activation"
    ] = True

    best[
        "promotion_response_band"
    ] = pd.cut(
        best[
            "promotion_response_propensity"
        ],
        bins=[
            -np.inf,
            0.50,
            0.60,
            0.70,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
        right=False,
    ).astype(str)

    return (
        best
        .sort_values(
            "promotion_response_propensity",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    scores: pd.DataFrame,
) -> None:

    print()
    print("=" * 82)
    print(
        "PROMOTION RESPONSE OPERATIONAL SCORING QA"
    )
    print("=" * 82)

    print(
        f"Scored Golden Customers       : "
        f"{len(scores):,}"
    )

    print(
        f"Unique Golden Customer IDs    : "
        f"{scores['golden_customer_id'].nunique():,}"
    )

    print(
        f"Duplicate Golden Customer IDs : "
        f"{scores['golden_customer_id'].duplicated().sum():,}"
    )

    print(
        f"Mean response propensity      : "
        f"{scores['promotion_response_propensity'].mean():.1%}"
    )

    print(
        f"Median response propensity    : "
        f"{scores['promotion_response_propensity'].median():.1%}"
    )

    print()
    print(
        "Recommended campaign channel:"
    )

    print(
        scores[
            "campaign_channel"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Recommended discount depth:"
    )

    print(
        scores[
            "discount_depth"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Promotion response bands:"
    )

    print(
        scores[
            "promotion_response_band"
        ]
        .value_counts()
        .to_string()
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
        "Loading current Golden Customer data..."
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    transactions[
        "transaction_date"
    ] = pd.to_datetime(
        transactions[
            "transaction_date"
        ]
    )

    # promotion_response_training.build_snapshot_features expects this
    # helper field to already exist on the transaction frame.
    transactions[
        "discounted_order"
    ] = (
        transactions[
            "discount_pct"
        ]
        .fillna(0)
        .gt(0)
    )

    priority = pd.read_parquet(
        PRIORITY_FILE
    )

    if not model_is_accepted():
        print(
            "Promotion-response model status: REJECTED / NOT ACCEPTED"
        )
        print(
            "Operational scoring skipped. NBA will use the transparent "
            "decision-engine fallback for Promotion."
        )

        empty_scores = pd.DataFrame(
            columns=[
                "golden_customer_id",
                "campaign_date",
                "campaign_channel",
                "offer_category",
                "discount_depth",
                "promotion_response_propensity",
                "promotion_response_propensity_raw",
                "offer_scenario_expected_contribution",
                "model_status",
                "model_candidate",
                "eligible_for_activation",
                "promotion_response_band",
            ]
        )

        empty_scores.to_parquet(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print(
            "Empty operational score artefact written:"
        )
        print(
            OUTPUT_FILE
        )
        return

    print(
        "Promotion-response model status: ACCEPTED"
    )
    print(
        "Scoring current promotion-response opportunities..."
    )

    scores = build_operational_scores(
        transactions,
        priority,
    )

    scores.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    run_qa(
        scores
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
