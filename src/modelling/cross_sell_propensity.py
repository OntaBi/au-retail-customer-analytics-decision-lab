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
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAINING_FILE = Path(
    "data/runtime/cross_sell_training.parquet"
)

MODEL_OUTPUT_FILE = Path(
    "data/runtime/cross_sell_model_scores.parquet"
)

MODEL_COMPARISON_FILE = Path(
    "data/runtime/cross_sell_model_comparison.parquet"
)

LIFT_FILE = Path(
    "data/runtime/cross_sell_model_lift.parquet"
)

COEFFICIENT_FILE = Path(
    "data/runtime/cross_sell_logistic_coefficients.parquet"
)

GOVERNANCE_FILE = Path(
    "data/runtime/cross_sell_model_governance.parquet"
)

METADATA_FILE = Path(
    "data/runtime/cross_sell_model_metadata.json"
)

TRAIN_START_DATE = pd.Timestamp("2024-07-31")
VALIDATION_DATE = pd.Timestamp("2025-10-31")
TEST_DATE = pd.Timestamp("2026-01-31")

TARGET = "entered_new_category_next_180d"

FORBIDDEN_FEATURES = {
    TARGET,
    "new_categories_next_180d",
    "future_180d_orders",
    "future_180d_sales",
    "future_180d_margin",
    "observation_date",
    "golden_customer_id",
    "cross_sell_eligible",
    "first_purchase_date",
    "last_purchase_date",
}

FEATURES = [
    "days_since_last_purchase",
    "purchase_tenure_days",
    "lifetime_orders",
    "lifetime_sales",
    "lifetime_margin",
    "lifetime_units",
    "avg_order_value",
    "avg_margin_per_order",
    "avg_discount_pct",
    "discounted_order_share",
    "channels_used",
    "categories_used",
    "margin_rate",
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
    "dominant_category_share",
    "dominant_category_sales_share",
    "category_concentration",
    "preferred_channel_share",
    "category_whitespace_count",
    "category_coverage_pct",
    "observation_month_sin",
    "observation_month_cos",
]

# Governance gates.
MIN_OOT_ROC_AUC = 0.60
MIN_PR_AUC_LIFT = 1.10
MIN_TOP_DECILE_LIFT = 1.20
MIN_TOP_QUINTILE_LIFT = 1.12
MAX_CALIBRATION_GAP = 0.05


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def add_time_features(
    data: pd.DataFrame,
) -> pd.DataFrame:

    result = data.copy()

    month = (
        result["observation_date"]
        .dt.month
        .astype(float)
    )

    result["observation_month_sin"] = np.sin(
        2 * np.pi * month / 12.0
    )

    result["observation_month_cos"] = np.cos(
        2 * np.pi * month / 12.0
    )

    return result


def load_training_data() -> pd.DataFrame:

    data = pd.read_parquet(
        TRAINING_FILE
    ).copy()

    required = (
        set(FEATURES)
        - {
            "observation_month_sin",
            "observation_month_cos",
        }
    ) | {
        TARGET,
        "observation_date",
        "golden_customer_id",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Cross-sell training data is missing required columns: "
            f"{sorted(missing)}"
        )

    overlap = (
        set(FEATURES)
        & FORBIDDEN_FEATURES
    )

    if overlap:
        raise ValueError(
            "Leakage-prone fields found in model features: "
            f"{sorted(overlap)}"
        )

    data["observation_date"] = pd.to_datetime(
        data["observation_date"]
    )

    return add_time_features(
        data
    )


def split_by_time(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    train = data.loc[
        data["observation_date"]
        .ge(TRAIN_START_DATE)
        & data["observation_date"]
        .lt(VALIDATION_DATE)
    ].copy()

    validation = data.loc[
        data["observation_date"]
        .eq(VALIDATION_DATE)
    ].copy()

    test = data.loc[
        data["observation_date"]
        .eq(TEST_DATE)
    ].copy()

    if train.empty:
        raise ValueError(
            "Cross-sell training split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Cross-sell validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Cross-sell out-of-time test split is empty."
        )

    return train, validation, test


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_logistic_model() -> Pipeline:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                FEATURES,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_gradient_boosting_model() -> Pipeline:

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(
                    strategy="median"
                ),
                FEATURES,
            )
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=40,
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


def build_lift_table(
    scored: pd.DataFrame,
    split_name: str,
    model_name: str,
    probability_type: str,
) -> pd.DataFrame:

    data = scored[
        [
            "golden_customer_id",
            TARGET,
            "predicted_probability",
        ]
    ].copy()

    data = (
        data
        .sort_values(
            "predicted_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    data["rank_pct"] = (
        (
            data.index
            + 1
        )
        / len(data)
    )

    data["decile"] = np.ceil(
        data["rank_pct"]
        * 10
    ).clip(
        1,
        10,
    ).astype(int)

    overall_rate = float(
        data[TARGET].mean()
    )

    lift = (
        data
        .groupby("decile")
        .agg(
            customers=(
                "golden_customer_id",
                "size",
            ),
            actual_cross_sell_rate=(
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

    lift["lift_vs_average"] = np.where(
        overall_rate > 0,
        lift["actual_cross_sell_rate"]
        / overall_rate,
        np.nan,
    )

    lift["split"] = split_name
    lift["model"] = model_name
    lift["probability_type"] = (
        probability_type
    )

    return lift


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

    overall_rate = (
        data["target"].mean()
    )

    top_rate = (
        data.head(n)[
            "target"
        ].mean()
    )

    if overall_rate <= 0:
        return np.nan

    return float(
        top_rate
        / overall_rate
    )


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

    validation_raw = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    calibrator = fit_sigmoid_calibrator(
        validation_raw,
        validation[TARGET],
    )

    validation_calibrated = calibrate_probability(
        calibrator,
        validation_raw,
    )

    test_raw = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    test_calibrated = calibrate_probability(
        calibrator,
        test_raw,
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
                frame[TARGET],
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
                        len(frame),
                    "top_decile_lift":
                        top_fraction_lift(
                            frame[TARGET],
                            probability,
                            0.10,
                        ),
                    "top_quintile_lift":
                        top_fraction_lift(
                            frame[TARGET],
                            probability,
                            0.20,
                        ),
                }
            )

            metric_rows.append(
                metrics
            )

            scored = frame[
                [
                    "golden_customer_id",
                    TARGET,
                ]
            ].copy()

            scored[
                "predicted_probability"
            ] = probability

            lift_rows.append(
                build_lift_table(
                    scored,
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
# Baseline
# ---------------------------------------------------------------------

def evaluate_constant_baseline(
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:

    baseline_probability = float(
        validation[TARGET].mean()
    )

    probability = np.repeat(
        baseline_probability,
        len(test),
    )

    metrics = evaluate_predictions(
        test[TARGET],
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
                len(test),
            "top_decile_lift":
                1.0,
            "top_quintile_lift":
                1.0,
        }
    )

    return metrics


# ---------------------------------------------------------------------
# Model selection and governance
# ---------------------------------------------------------------------

def select_candidate(
    metrics: pd.DataFrame,
) -> str:

    validation = metrics.loc[
        metrics["split"].eq(
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
        .reset_index(drop=True)
    )

    top = ranked.iloc[0]

    logistic = ranked.loc[
        ranked["model"]
        .eq(
            "LogisticRegression"
        )
    ]

    if (
        top["model"]
        == "HistGradientBoosting"
        and not logistic.empty
    ):
        logistic_row = (
            logistic.iloc[0]
        )

        if (
            top["roc_auc"]
            - logistic_row["roc_auc"]
            < 0.02
            and top["top_decile_lift"]
            - logistic_row["top_decile_lift"]
            < 0.08
        ):
            return "LogisticRegression"

    return str(
        top["model"]
    )


def evaluate_governance(
    champion_name: str,
    metrics: pd.DataFrame,
) -> pd.DataFrame:

    champion_test = metrics.loc[
        metrics["model"].eq(
            champion_name
        )
        & metrics["split"].eq(
            "test"
        )
        & metrics[
            "probability_type"
        ].eq(
            "calibrated"
        )
    ].iloc[0]

    baseline = metrics.loc[
        metrics["model"].eq(
            "RecentBaseRate"
        )
        & metrics["split"].eq(
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
        ] > 0
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
# Explainability
# ---------------------------------------------------------------------

def extract_logistic_coefficients(
    model: Pipeline,
) -> pd.DataFrame:

    logistic = model.named_steps[
        "model"
    ]

    coefficients = (
        logistic.coef_[0]
    )

    result = pd.DataFrame(
        {
            "feature":
                FEATURES,
            "coefficient":
                coefficients,
            "absolute_coefficient":
                np.abs(
                    coefficients
                ),
        }
    )

    result["direction"] = np.where(
        result["coefficient"] >= 0,
        "Higher cross-sell propensity",
        "Lower cross-sell propensity",
    )

    return (
        result
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------
# Governed score output
# ---------------------------------------------------------------------

def build_test_scores(
    test: pd.DataFrame,
    champion: Pipeline,
    calibrator: LogisticRegression,
    model_status: str,
) -> pd.DataFrame:

    raw_probability = champion.predict_proba(
        test[FEATURES]
    )[:, 1]

    calibrated_probability = (
        calibrate_probability(
            calibrator,
            raw_probability,
        )
    )

    result = test[
        [
            "golden_customer_id",
            "observation_date",
            TARGET,
            "categories_used",
            "category_whitespace_count",
            "category_coverage_pct",
            "dominant_category_share",
            "category_concentration",
            "lifetime_orders",
            "orders_prior_180d",
            "orders_prior_365d",
        ]
    ].copy()

    result[
        "cross_sell_propensity_180d_raw"
    ] = raw_probability

    result[
        "cross_sell_propensity_180d"
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
        "cross_sell_propensity_band"
    ] = pd.cut(
        result[
            "cross_sell_propensity_180d"
        ],
        bins=[
            -np.inf,
            0.25,
            0.50,
            0.75,
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
            "cross_sell_propensity_180d",
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
) -> None:

    print()
    print("=" * 86)
    print("180-DAY CROSS-SELL PROPENSITY MODEL — GOVERNED")
    print("=" * 86)

    print(
        f"Rolling training window : "
        f"{TRAIN_START_DATE.date()} to "
        f"{(VALIDATION_DATE - pd.Timedelta(days=1)).date()}"
    )

    print(
        f"Validation snapshot     : "
        f"{VALIDATION_DATE.date()}"
    )

    print(
        f"OOT test snapshot       : "
        f"{TEST_DATE.date()}"
    )

    print()
    print("MODEL COMPARISON")
    print("-" * 86)

    display = metrics.copy()

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
        display[column] = (
            display[column]
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
    print("-" * 86)

    governance_display = (
        governance.copy()
    )

    governance_display[
        "actual"
    ] = (
        governance_display[
            "actual"
        ]
        .map(
            lambda x:
                f"{x:.4f}"
        )
    )

    governance_display[
        "threshold"
    ] = (
        governance_display[
            "threshold"
        ]
        .map(
            lambda x:
                f"{x:.4f}"
        )
    )

    print(
        governance_display
        .to_string(
            index=False
        )
    )

    status = (
        "ACCEPTED"
        if governance["pass"].all()
        else "REJECTED"
    )

    print()
    print(
        f"Final model status: "
        f"{status}"
    )

    champion_lift = (
        lift.loc[
            lift["model"].eq(
                champion_name
            )
            & lift["split"].eq(
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
        print("-" * 86)

        champion_lift[
            "actual_cross_sell_rate"
        ] = (
            champion_lift[
                "actual_cross_sell_rate"
            ]
            .map(
                lambda x:
                    f"{x:.1%}"
            )
        )

        champion_lift[
            "avg_predicted_probability"
        ] = (
            champion_lift[
                "avg_predicted_probability"
            ]
            .map(
                lambda x:
                    f"{x:.1%}"
            )
        )

        champion_lift[
            "lift_vs_average"
        ] = (
            champion_lift[
                "lift_vs_average"
            ]
            .map(
                lambda x:
                    f"{x:.2f}x"
            )
        )

        print(
            champion_lift[
                [
                    "decile",
                    "customers",
                    "actual_cross_sell_rate",
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
        "Loading cross-sell propensity training data..."
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
        f"Training rows: {len(train):,}"
    )

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Out-of-time test rows: {len(test):,}"
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

    for model_name, model in models.items():

        print(
            f"\nFitting {model_name}..."
        )

        model.fit(
            train[FEATURES],
            train[TARGET],
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
        evaluate_constant_baseline(
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

    champion_name = select_candidate(
        metrics_df
    )

    governance = evaluate_governance(
        champion_name,
        metrics_df,
    )

    model_status = (
        "ACCEPTED"
        if governance["pass"].all()
        else "REJECTED"
    )

    champion = fitted_models[
        champion_name
    ]

    champion_calibrator = calibrators[
        champion_name
    ]

    governed_scores = build_test_scores(
        test,
        champion,
        champion_calibrator,
        model_status,
    )

    coefficients = extract_logistic_coefficients(
        fitted_models[
            "LogisticRegression"
        ]
    )

    metrics_df.to_parquet(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    lift_df.to_parquet(
        LIFT_FILE,
        index=False,
    )

    coefficients.to_parquet(
        COEFFICIENT_FILE,
        index=False,
    )

    governance.to_parquet(
        GOVERNANCE_FILE,
        index=False,
    )

    governed_scores.to_parquet(
        MODEL_OUTPUT_FILE,
        index=False,
    )

    champion_test = (
        metrics_df.loc[
            metrics_df["model"].eq(
                champion_name
            )
            & metrics_df["split"].eq(
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
            "180-Day Cross-Sell Propensity",
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
        "feature_count":
            len(FEATURES),
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
        "activation_allowed":
            bool(
                model_status
                == "ACCEPTED"
            ),
        "target":
            TARGET,
        "notes":
            (
                "Predicts whether an eligible Golden Customer enters "
                "a previously unpurchased category within 180 days. "
                "Scores must not be used for activation when model_status "
                "is REJECTED."
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
    )

    print()
    print(
        "Files created:"
    )

    print(MODEL_OUTPUT_FILE)
    print(MODEL_COMPARISON_FILE)
    print(LIFT_FILE)
    print(COEFFICIENT_FILE)
    print(GOVERNANCE_FILE)
    print(METADATA_FILE)


if __name__ == "__main__":
    main()
