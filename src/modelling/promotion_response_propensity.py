from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAINING_FILE = Path(
    "data/runtime/promotion_response_training.parquet"
)

MODEL_OUTPUT_FILE = Path(
    "data/runtime/promotion_response_model_scores.parquet"
)

MODEL_COMPARISON_FILE = Path(
    "data/runtime/promotion_response_model_comparison.parquet"
)

LIFT_FILE = Path(
    "data/runtime/promotion_response_model_lift.parquet"
)

GOVERNANCE_FILE = Path(
    "data/runtime/promotion_response_model_governance.parquet"
)

METADATA_FILE = Path(
    "data/runtime/promotion_response_model_metadata.json"
)

# Model only treated customers: response conditional on actually receiving an offer.
TRAIN_START_DATE = pd.Timestamp("2024-08-15")
VALIDATION_DATE = pd.Timestamp("2026-02-15")
TEST_DATE = pd.Timestamp("2026-05-15")

TARGET = "responded"

NUMERIC_FEATURES = [
    "discount_depth",
    "days_since_last_purchase",
    "customer_tenure_days",
    "lifetime_orders",
    "lifetime_sales",
    "lifetime_margin",
    "average_order_value",
    "average_discount_pct",
    "discounted_order_share",
    "lifetime_categories",
    "lifetime_channels",
    "lifetime_margin_rate",
    "orders_prior_180d",
    "sales_prior_180d",
    "margin_prior_180d",
    "categories_prior_180d",
    "channels_prior_180d",
    "avg_discount_prior_180d",
    "discounted_order_share_prior_180d",
    "offer_category_prior_orders",
    "offer_category_prior_sales",
    "offer_category_order_share",
    "offer_category_seen_before",
    "recent_margin_rate",
    "orders_per_30d_tenure",
    "campaign_month_sin",
    "campaign_month_cos",
]

CATEGORICAL_FEATURES = [
    "campaign_channel",
    "offer_category",
]

FORBIDDEN_FEATURES = {
    TARGET,
    "treatment_flag",
    "response_probability_used",
    "incremental_probability_truth",
    "natural_purchase_in_window",
    "natural_category_purchase_in_window",
    "campaign_date",
    "campaign_id",
    "exposure_id",
    "golden_customer_id",
    "response_window_days",
    "preferred_channel",
    "campaign_channel_matches_preference",
}

MIN_OOT_ROC_AUC = 0.60
MIN_PR_AUC_LIFT = 1.08
MIN_TOP_DECILE_LIFT = 1.15
MIN_TOP_QUINTILE_LIFT = 1.10
MAX_CALIBRATION_GAP = 0.05


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    month = (
        result["campaign_date"]
        .dt.month
        .astype(float)
    )

    result["campaign_month_sin"] = np.sin(
        2 * np.pi * month / 12.0
    )

    result["campaign_month_cos"] = np.cos(
        2 * np.pi * month / 12.0
    )

    return result


def load_training_data() -> pd.DataFrame:

    data = pd.read_parquet(
        TRAINING_FILE
    ).copy()

    data["campaign_date"] = pd.to_datetime(
        data["campaign_date"]
    )

    # Response propensity is modelled among customers actually exposed
    # to a promotion. Control rows remain available as a benchmark for
    # incrementality, but are not mixed into the response model.
    data = data.loc[
        data["treatment_flag"]
        .eq(1)
    ].copy()

    required = (
        set(NUMERIC_FEATURES)
        - {
            "campaign_month_sin",
            "campaign_month_cos",
        }
    ) | set(CATEGORICAL_FEATURES) | {
        TARGET,
        "campaign_date",
        "campaign_id",
        "exposure_id",
        "golden_customer_id",
    }

    missing = required - set(
        data.columns
    )

    if missing:
        raise ValueError(
            "Promotion response training data is missing required columns: "
            f"{sorted(missing)}"
        )

    overlap = (
        (
            set(NUMERIC_FEATURES)
            | set(CATEGORICAL_FEATURES)
        )
        & FORBIDDEN_FEATURES
    )

    if overlap:
        raise ValueError(
            "Leakage-prone fields found in promotion response features: "
            f"{sorted(overlap)}"
        )

    data = add_time_features(
        data
    )

    return data


def split_by_time(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    train = data.loc[
        data["campaign_date"]
        .ge(TRAIN_START_DATE)
        & data["campaign_date"]
        .lt(VALIDATION_DATE)
    ].copy()

    validation = data.loc[
        data["campaign_date"]
        .eq(VALIDATION_DATE)
    ].copy()

    test = data.loc[
        data["campaign_date"]
        .eq(TEST_DATE)
    ].copy()

    if train.empty:
        raise ValueError(
            "Promotion response training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Promotion response validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Promotion response OOT test split is empty."
        )

    return train, validation, test


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def build_preprocessor(
    scale_numeric: bool,
) -> ColumnTransformer:

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_logistic_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    scale_numeric=True
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_gradient_boosting_model() -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    scale_numeric=False
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=50,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def probability_to_logit(
    probability: np.ndarray,
) -> np.ndarray:

    p = np.clip(
        probability,
        1e-5,
        1 - 1e-5,
    )

    return np.log(
        p / (1 - p)
    )


def fit_sigmoid_calibrator(
    raw_probability: np.ndarray,
    y_true: pd.Series,
) -> LogisticRegression:

    x = probability_to_logit(
        raw_probability
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    calibrator.fit(
        x,
        y_true,
    )

    return calibrator


def calibrate_probability(
    calibrator: LogisticRegression,
    raw_probability: np.ndarray,
) -> np.ndarray:

    x = probability_to_logit(
        raw_probability
    ).reshape(-1, 1)

    return calibrator.predict_proba(
        x
    )[:, 1]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true: pd.Series,
    probability: np.ndarray,
) -> dict:

    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6,
    )

    base_rate = float(
        np.mean(y_true)
    )

    mean_predicted = float(
        np.mean(probability)
    )

    return {
        "roc_auc":
            roc_auc_score(
                y_true,
                probability,
            ),
        "pr_auc":
            average_precision_score(
                y_true,
                probability,
            ),
        "log_loss":
            log_loss(
                y_true,
                probability,
            ),
        "brier_score":
            brier_score_loss(
                y_true,
                probability,
            ),
        "base_rate":
            base_rate,
        "mean_predicted_probability":
            mean_predicted,
        "calibration_gap":
            abs(
                mean_predicted
                - base_rate
            ),
    }


def top_fraction_lift(
    y_true: pd.Series,
    probability: np.ndarray,
    fraction: float,
) -> float:

    data = pd.DataFrame(
        {
            "target":
                np.asarray(y_true),
            "probability":
                probability,
        }
    )

    data = data.sort_values(
        "probability",
        ascending=False,
    )

    n = max(
        1,
        int(
            np.ceil(
                len(data)
                * fraction
            )
        ),
    )

    overall_rate = float(
        data["target"].mean()
    )

    top_rate = float(
        data.head(n)[
            "target"
        ].mean()
    )

    if overall_rate <= 0:
        return np.nan

    return (
        top_rate
        / overall_rate
    )


def build_lift_table(
    frame: pd.DataFrame,
    probability: np.ndarray,
    split_name: str,
    model_name: str,
    probability_type: str,
) -> pd.DataFrame:

    data = frame[
        [
            "exposure_id",
            TARGET,
        ]
    ].copy()

    data[
        "predicted_probability"
    ] = probability

    data = (
        data
        .sort_values(
            "predicted_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    data[
        "rank_pct"
    ] = (
        data.index
        + 1
    ) / len(
        data
    )

    data[
        "decile"
    ] = np.ceil(
        data[
            "rank_pct"
        ]
        * 10
    ).clip(
        1,
        10,
    ).astype(
        int
    )

    overall_rate = float(
        data[TARGET].mean()
    )

    lift = (
        data
        .groupby(
            "decile"
        )
        .agg(
            exposures=(
                "exposure_id",
                "size",
            ),
            actual_response_rate=(
                TARGET,
                "mean",
            ),
            avg_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    lift[
        "lift_vs_average"
    ] = np.where(
        overall_rate > 0,
        lift[
            "actual_response_rate"
        ] / overall_rate,
        np.nan,
    )

    lift[
        "split"
    ] = split_name

    lift[
        "model"
    ] = model_name

    lift[
        "probability_type"
    ] = probability_type

    return lift


def score_model(
    model: Pipeline,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
) -> tuple[
    list[dict],
    list[pd.DataFrame],
    LogisticRegression,
]:

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    validation_raw = model.predict_proba(
        validation[
            feature_columns
        ]
    )[:, 1]

    calibrator = fit_sigmoid_calibrator(
        validation_raw,
        validation[
            TARGET
        ],
    )

    validation_calibrated = (
        calibrate_probability(
            calibrator,
            validation_raw,
        )
    )

    test_raw = model.predict_proba(
        test[
            feature_columns
        ]
    )[:, 1]

    test_calibrated = (
        calibrate_probability(
            calibrator,
            test_raw,
        )
    )

    metric_rows = []
    lift_rows = []

    for (
        split_name,
        frame,
        raw_probability,
        calibrated_probability,
    ) in [
        (
            "validation",
            validation,
            validation_raw,
            validation_calibrated,
        ),
        (
            "test",
            test,
            test_raw,
            test_calibrated,
        ),
    ]:

        for (
            probability_type,
            probability,
        ) in [
            (
                "raw",
                raw_probability,
            ),
            (
                "calibrated",
                calibrated_probability,
            ),
        ]:

            metrics = evaluate_predictions(
                frame[
                    TARGET
                ],
                probability,
            )

            metrics.update(
                {
                    "model":
                        model_name,
                    "split":
                        split_name,
                    "probability_type":
                        probability_type,
                    "rows":
                        len(
                            frame
                        ),
                    "top_decile_lift":
                        top_fraction_lift(
                            frame[
                                TARGET
                            ],
                            probability,
                            0.10,
                        ),
                    "top_quintile_lift":
                        top_fraction_lift(
                            frame[
                                TARGET
                            ],
                            probability,
                            0.20,
                        ),
                }
            )

            metric_rows.append(
                metrics
            )

            lift_rows.append(
                build_lift_table(
                    frame,
                    probability,
                    split_name,
                    model_name,
                    probability_type,
                )
            )

    return (
        metric_rows,
        lift_rows,
        calibrator,
    )


# ---------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------

def evaluate_recent_base_rate(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:

    baseline_probability = float(
        validation[
            TARGET
        ].mean()
    )

    probability = np.repeat(
        baseline_probability,
        len(
            test
        ),
    )

    metrics = evaluate_predictions(
        test[
            TARGET
        ],
        probability,
    )

    metrics.update(
        {
            "model":
                "RecentBaseRate",
            "split":
                "test",
            "probability_type":
                "baseline",
            "rows":
                len(
                    test
                ),
            "top_decile_lift":
                1.0,
            "top_quintile_lift":
                1.0,
        }
    )

    return metrics


def get_control_benchmark(
    all_training: pd.DataFrame,
    campaign_date: pd.Timestamp,
) -> dict:

    control = all_training.loc[
        all_training[
            "campaign_date"
        ]
        .eq(
            campaign_date
        )
        & all_training[
            "treatment_flag"
        ]
        .eq(0)
    ].copy()

    if control.empty:
        return {
            "control_rows":
                0,
            "control_response_rate":
                np.nan,
        }

    return {
        "control_rows":
            len(
                control
            ),
        "control_response_rate":
            float(
                control[
                    TARGET
                ].mean()
            ),
    }


# ---------------------------------------------------------------------
# Candidate selection + governance
# ---------------------------------------------------------------------

def select_candidate(
    metrics: pd.DataFrame,
) -> str:

    validation = metrics.loc[
        metrics[
            "split"
        ].eq(
            "validation"
        )
        & metrics[
            "probability_type"
        ].eq(
            "raw"
        )
        & metrics[
            "model"
        ].isin(
            [
                "LogisticRegression",
                "HistGradientBoosting",
            ]
        )
    ].copy()

    ranked = (
        validation
        .sort_values(
            [
                "roc_auc",
                "pr_auc",
                "top_decile_lift",
                "brier_score",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    top = ranked.iloc[0]

    logistic = ranked.loc[
        ranked[
            "model"
        ].eq(
            "LogisticRegression"
        )
    ]

    if (
        top[
            "model"
        ]
        == "HistGradientBoosting"
        and not logistic.empty
    ):
        logistic_row = (
            logistic.iloc[0]
        )

        if (
            top[
                "roc_auc"
            ]
            - logistic_row[
                "roc_auc"
            ]
            < 0.02
            and top[
                "top_decile_lift"
            ]
            - logistic_row[
                "top_decile_lift"
            ]
            < 0.05
        ):
            return (
                "LogisticRegression"
            )

    return str(
        top[
            "model"
        ]
    )


def evaluate_governance(
    champion_name: str,
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    champion_test = metrics.loc[
        metrics[
            "model"
        ].eq(
            champion_name
        )
        & metrics[
            "split"
        ].eq(
            "test"
        )
        & metrics[
            "probability_type"
        ].eq(
            "calibrated"
        )
    ].iloc[0]

    baseline = metrics.loc[
        metrics[
            "model"
        ].eq(
            "RecentBaseRate"
        )
        & metrics[
            "split"
        ].eq(
            "test"
        )
    ].iloc[0]

    pr_auc_lift = (
        float(
            champion_test[
                "pr_auc"
            ]
        )
        / float(
            champion_test[
                "base_rate"
            ]
        )
        if champion_test[
            "base_rate"
        ]
        > 0
        else np.nan
    )

    checks = [
        {
            "gate":
                "OOT ROC-AUC",
            "actual":
                float(
                    champion_test[
                        "roc_auc"
                    ]
                ),
            "threshold":
                MIN_OOT_ROC_AUC,
            "pass":
                float(
                    champion_test[
                        "roc_auc"
                    ]
                )
                >= MIN_OOT_ROC_AUC,
        },
        {
            "gate":
                "PR-AUC lift vs base rate",
            "actual":
                pr_auc_lift,
            "threshold":
                MIN_PR_AUC_LIFT,
            "pass":
                pr_auc_lift
                >= MIN_PR_AUC_LIFT,
        },
        {
            "gate":
                "Top-decile lift",
            "actual":
                float(
                    champion_test[
                        "top_decile_lift"
                    ]
                ),
            "threshold":
                MIN_TOP_DECILE_LIFT,
            "pass":
                float(
                    champion_test[
                        "top_decile_lift"
                    ]
                )
                >= MIN_TOP_DECILE_LIFT,
        },
        {
            "gate":
                "Top-quintile lift",
            "actual":
                float(
                    champion_test[
                        "top_quintile_lift"
                    ]
                ),
            "threshold":
                MIN_TOP_QUINTILE_LIFT,
            "pass":
                float(
                    champion_test[
                        "top_quintile_lift"
                    ]
                )
                >= MIN_TOP_QUINTILE_LIFT,
        },
        {
            "gate":
                "Calibration gap",
            "actual":
                float(
                    champion_test[
                        "calibration_gap"
                    ]
                ),
            "threshold":
                MAX_CALIBRATION_GAP,
            "pass":
                float(
                    champion_test[
                        "calibration_gap"
                    ]
                )
                <= MAX_CALIBRATION_GAP,
        },
        {
            "gate":
                "Brier score vs recent-base-rate baseline",
            "actual":
                float(
                    champion_test[
                        "brier_score"
                    ]
                ),
            "threshold":
                float(
                    baseline[
                        "brier_score"
                    ]
                ),
            "pass":
                float(
                    champion_test[
                        "brier_score"
                    ]
                )
                <= float(
                    baseline[
                        "brier_score"
                    ]
                ),
        },
    ]

    return pd.DataFrame(
        checks
    )


# ---------------------------------------------------------------------
# Output scoring
# ---------------------------------------------------------------------

def build_test_scores(
    test: pd.DataFrame,
    champion: Pipeline,
    calibrator: LogisticRegression,
    model_status: str,
) -> pd.DataFrame:

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    raw_probability = (
        champion
        .predict_proba(
            test[
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

    result = test[
        [
            "exposure_id",
            "campaign_id",
            "campaign_date",
            "golden_customer_id",
            "campaign_channel",
            "offer_category",
            "discount_depth",
            TARGET,
        ]
    ].copy()

    result[
        "promotion_response_propensity_raw"
    ] = raw_probability

    result[
        "promotion_response_propensity"
    ] = calibrated_probability

    result[
        "model_status"
    ] = model_status

    result[
        "eligible_for_activation"
    ] = (
        model_status
        == "ACCEPTED"
    )

    result[
        "promotion_response_band"
    ] = pd.cut(
        result[
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
        result
        .sort_values(
            "promotion_response_propensity",
            ascending=False,
        )
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_report(
    metrics: pd.DataFrame,
    champion_name: str,
    governance: pd.DataFrame,
    lift: pd.DataFrame,
    control_benchmark: dict,
) -> None:

    print()
    print("=" * 90)
    print("30-DAY PROMOTION RESPONSE PROPENSITY MODEL — GOVERNED")
    print("=" * 90)

    print(
        f"Rolling training window : "
        f"{TRAIN_START_DATE.date()} to "
        f"{(VALIDATION_DATE - pd.Timedelta(days=1)).date()}"
    )

    print(
        f"Validation campaign     : "
        f"{VALIDATION_DATE.date()}"
    )

    print(
        f"OOT test campaign       : "
        f"{TEST_DATE.date()}"
    )

    print()
    print("MODEL COMPARISON")
    print("-" * 90)

    display = (
        metrics.copy()
    )

    for column in [
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "base_rate",
        "mean_predicted_probability",
        "calibration_gap",
        "top_decile_lift",
        "top_quintile_lift",
    ]:
        display[
            column
        ] = (
            display[
                column
            ]
            .map(
                lambda x:
                    f"{x:.4f}"
            )
        )

    print(
        display[
            [
                "model",
                "split",
                "probability_type",
                "rows",
                "roc_auc",
                "pr_auc",
                "log_loss",
                "brier_score",
                "base_rate",
                "mean_predicted_probability",
                "calibration_gap",
                "top_decile_lift",
                "top_quintile_lift",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected candidate: "
        f"{champion_name}"
    )

    print()
    print("MODEL GOVERNANCE")
    print("-" * 90)

    governance_display = (
        governance.copy()
    )

    governance_display[
        "actual"
    ] = governance_display[
        "actual"
    ].map(
        lambda x:
            f"{x:.4f}"
    )

    governance_display[
        "threshold"
    ] = governance_display[
        "threshold"
    ].map(
        lambda x:
            f"{x:.4f}"
    )

    print(
        governance_display.to_string(
            index=False
        )
    )

    status = (
        "ACCEPTED"
        if governance[
            "pass"
        ].all()
        else "REJECTED"
    )

    print()
    print(
        f"Final model status: "
        f"{status}"
    )

    print()
    print("CONTROL BENCHMARK")
    print("-" * 90)

    print(
        f"OOT control rows          : "
        f"{control_benchmark['control_rows']:,}"
    )

    control_rate = (
        control_benchmark[
            "control_response_rate"
        ]
    )

    if pd.notna(
        control_rate
    ):
        print(
            f"OOT control response rate : "
            f"{control_rate:.1%}"
        )

    champion_lift = (
        lift.loc[
            lift[
                "model"
            ].eq(
                champion_name
            )
            & lift[
                "split"
            ].eq(
                "test"
            )
            & lift[
                "probability_type"
            ].eq(
                "calibrated"
            )
        ]
        .copy()
    )

    if not champion_lift.empty:
        print()
        print(
            "OOT CALIBRATED DECILE LIFT"
        )
        print("-" * 90)

        champion_lift[
            "actual_response_rate"
        ] = champion_lift[
            "actual_response_rate"
        ].map(
            lambda x:
                f"{x:.1%}"
        )

        champion_lift[
            "avg_predicted_probability"
        ] = champion_lift[
            "avg_predicted_probability"
        ].map(
            lambda x:
                f"{x:.1%}"
        )

        champion_lift[
            "lift_vs_average"
        ] = champion_lift[
            "lift_vs_average"
        ].map(
            lambda x:
                f"{x:.2f}x"
        )

        print(
            champion_lift[
                [
                    "decile",
                    "exposures",
                    "actual_response_rate",
                    "avg_predicted_probability",
                    "lift_vs_average",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    MODEL_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading promotion-response training data..."
    )

    all_training = pd.read_parquet(
        TRAINING_FILE
    ).copy()

    all_training[
        "campaign_date"
    ] = pd.to_datetime(
        all_training[
            "campaign_date"
        ]
    )

    data = load_training_data()

    (
        train,
        validation,
        test,
    ) = split_by_time(
        data
    )

    print(
        f"Training treated rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation treated rows: "
        f"{len(validation):,}"
    )

    print(
        f"OOT test treated rows: "
        f"{len(test):,}"
    )

    models = {
        "LogisticRegression":
            build_logistic_model(),
        "HistGradientBoosting":
            build_gradient_boosting_model(),
    }

    metric_rows = []
    lift_rows = []
    fitted_models = {}
    calibrators = {}

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    for model_name, model in models.items():

        print(
            f"\nFitting {model_name}..."
        )

        model.fit(
            train[
                feature_columns
            ],
            train[
                TARGET
            ],
        )

        fitted_models[
            model_name
        ] = model

        (
            model_metrics,
            model_lift,
            calibrator,
        ) = score_model(
            model,
            validation,
            test,
            model_name,
        )

        metric_rows.extend(
            model_metrics
        )

        lift_rows.extend(
            model_lift
        )

        calibrators[
            model_name
        ] = calibrator

    metric_rows.append(
        evaluate_recent_base_rate(
            validation,
            test,
        )
    )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    lift_df = pd.concat(
        lift_rows,
        ignore_index=True,
    )

    champion_name = (
        select_candidate(
            metrics_df
        )
    )

    governance = (
        evaluate_governance(
            champion_name,
            metrics_df,
        )
    )

    model_status = (
        "ACCEPTED"
        if governance[
            "pass"
        ].all()
        else "REJECTED"
    )

    champion = fitted_models[
        champion_name
    ]

    calibrator = calibrators[
        champion_name
    ]

    scores = build_test_scores(
        test,
        champion,
        calibrator,
        model_status,
    )

    control_benchmark = (
        get_control_benchmark(
            all_training,
            TEST_DATE,
        )
    )

    metrics_df.to_parquet(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    lift_df.to_parquet(
        LIFT_FILE,
        index=False,
    )

    governance.to_parquet(
        GOVERNANCE_FILE,
        index=False,
    )

    scores.to_parquet(
        MODEL_OUTPUT_FILE,
        index=False,
    )

    champion_test = (
        metrics_df.loc[
            metrics_df[
                "model"
            ].eq(
                champion_name
            )
            & metrics_df[
                "split"
            ].eq(
                "test"
            )
            & metrics_df[
                "probability_type"
            ].eq(
                "calibrated"
            )
        ]
        .iloc[0]
    )

    metadata = {
        "model_name":
            "30-Day Promotion Response Propensity",
        "selected_candidate":
            champion_name,
        "model_status":
            model_status,
        "training_start":
            str(
                TRAIN_START_DATE.date()
            ),
        "training_end":
            str(
                (
                    VALIDATION_DATE
                    - pd.Timedelta(days=1)
                ).date()
            ),
        "validation_date":
            str(
                VALIDATION_DATE.date()
            ),
        "test_date":
            str(
                TEST_DATE.date()
            ),
        "treated_only_model":
            True,
        "test_roc_auc":
            float(
                champion_test[
                    "roc_auc"
                ]
            ),
        "test_pr_auc":
            float(
                champion_test[
                    "pr_auc"
                ]
            ),
        "test_brier_score":
            float(
                champion_test[
                    "brier_score"
                ]
            ),
        "test_top_decile_lift":
            float(
                champion_test[
                    "top_decile_lift"
                ]
            ),
        "test_top_quintile_lift":
            float(
                champion_test[
                    "top_quintile_lift"
                ]
            ),
        "oot_control_response_rate":
            (
                float(
                    control_benchmark[
                        "control_response_rate"
                    ]
                )
                if pd.notna(
                    control_benchmark[
                        "control_response_rate"
                    ]
                )
                else None
            ),
        "activation_allowed":
            bool(
                model_status
                == "ACCEPTED"
            ),
        "notes":
            (
                "Predicts 30-day response among treated customers who "
                "received a promotion. Control rows are retained separately "
                "to benchmark natural response and support future uplift "
                "modelling. This is response propensity, not causal uplift."
            ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_report(
        metrics_df,
        champion_name,
        governance,
        lift_df,
        control_benchmark,
    )

    print()
    print(
        "Files created:"
    )

    print(
        MODEL_OUTPUT_FILE
    )

    print(
        MODEL_COMPARISON_FILE
    )

    print(
        LIFT_FILE
    )

    print(
        GOVERNANCE_FILE
    )

    print(
        METADATA_FILE
    )


if __name__ == "__main__":
    main()
