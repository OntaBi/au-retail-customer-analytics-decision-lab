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
    "data/generated/campaign_exposures.parquet"
)

GROUND_TRUTH_FILE = Path(
    "data/generated/campaign_exposure_ground_truth.parquet"
)

RNG_SEED = 42

CAMPAIGN_DATES = pd.to_datetime(
    [
        "2024-02-15",
        "2024-05-15",
        "2024-08-15",
        "2024-11-15",
        "2025-02-15",
        "2025-05-15",
        "2025-08-15",
        "2025-11-15",
        "2026-02-15",
        "2026-05-15",
    ]
)

RESPONSE_WINDOW_DAYS = 30

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

TREATMENT_SHARE = 0.80

MIN_PRIOR_ORDERS = 2
MIN_PURCHASE_TENURE_DAYS = 90


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sigmoid(x: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(
        x,
        dtype=float,
    )

    return 1.0 / (
        1.0
        + np.exp(
            -values
        )
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator / denominator.replace(
        0,
        np.nan,
    )


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
        "discount_pct",
        "channel",
        "category",
    }

    missing = required - set(
        tx.columns
    )

    if missing:
        raise ValueError(
            "Golden customer transactions are missing "
            f"required columns: {sorted(missing)}"
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


def build_customer_snapshot(
    transactions: pd.DataFrame,
    campaign_date: pd.Timestamp,
) -> pd.DataFrame:

    history = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .lt(
            campaign_date
        )
    ].copy()

    if history.empty:
        return pd.DataFrame()

    snapshot = (
        history
        .groupby(
            "golden_customer_id"
        )
        .agg(
            first_purchase_date=(
                "transaction_date",
                "min",
            ),
            last_purchase_date=(
                "transaction_date",
                "max",
            ),
            lifetime_orders=(
                "order_id",
                "nunique",
            ),
            lifetime_sales=(
                "net_sales",
                "sum",
            ),
            lifetime_margin=(
                "gross_margin",
                "sum",
            ),
            average_order_value=(
                "net_sales",
                "mean",
            ),
            average_discount_pct=(
                "discount_pct",
                "mean",
            ),
            discounted_order_share=(
                "discounted_order",
                "mean",
            ),
            categories_used=(
                "category",
                "nunique",
            ),
            channels_used=(
                "channel",
                "nunique",
            ),
        )
        .reset_index()
    )

    snapshot[
        "days_since_last_purchase"
    ] = (
        campaign_date
        - snapshot[
            "last_purchase_date"
        ]
    ).dt.days

    snapshot[
        "purchase_tenure_days"
    ] = (
        campaign_date
        - snapshot[
            "first_purchase_date"
        ]
    ).dt.days

    snapshot[
        "margin_rate"
    ] = safe_divide(
        snapshot[
            "lifetime_margin"
        ],
        snapshot[
            "lifetime_sales"
        ],
    )

    start_180 = (
        campaign_date
        - pd.Timedelta(
            days=179
        )
    )

    recent = history.loc[
        history[
            "transaction_date"
        ]
        .between(
            start_180,
            campaign_date
            - pd.Timedelta(days=1),
        )
    ].copy()

    recent_agg = (
        recent
        .groupby(
            "golden_customer_id"
        )
        .agg(
            orders_prior_180d=(
                "order_id",
                "nunique",
            ),
            sales_prior_180d=(
                "net_sales",
                "sum",
            ),
            margin_prior_180d=(
                "gross_margin",
                "sum",
            ),
            categories_prior_180d=(
                "category",
                "nunique",
            ),
        )
        .reset_index()
    )

    snapshot = snapshot.merge(
        recent_agg,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "orders_prior_180d",
        "sales_prior_180d",
        "margin_prior_180d",
        "categories_prior_180d",
    ]:
        snapshot[column] = (
            snapshot[column]
            .fillna(0)
        )

    snapshot = snapshot.loc[
        snapshot[
            "lifetime_orders"
        ]
        .ge(
            MIN_PRIOR_ORDERS
        )
        & snapshot[
            "purchase_tenure_days"
        ]
        .ge(
            MIN_PURCHASE_TENURE_DAYS
        )
    ].copy()

    return snapshot


def build_preferred_category(
    transactions: pd.DataFrame,
    campaign_date: pd.Timestamp,
) -> pd.Series:

    history = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .lt(
            campaign_date
        )
    ].copy()

    category_profile = (
        history
        .groupby(
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
        .set_index(
            "golden_customer_id"
        )[
            "category"
        ]
    )

    return category_profile


def observed_category_response(
    transactions: pd.DataFrame,
    campaign_date: pd.Timestamp,
    exposure_frame: pd.DataFrame,
) -> pd.DataFrame:

    response_end = (
        campaign_date
        + pd.Timedelta(
            days=RESPONSE_WINDOW_DAYS
        )
    )

    future = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .gt(
            campaign_date
        )
        & transactions[
            "transaction_date"
        ]
        .le(
            response_end
        )
    ].copy()

    if future.empty:
        result = exposure_frame[
            [
                "exposure_id",
            ]
        ].copy()

        result[
            "natural_purchase_in_window"
        ] = 0

        result[
            "natural_category_purchase_in_window"
        ] = 0

        return result

    any_purchase = (
        future
        .groupby(
            "golden_customer_id"
        )[
            "order_id"
        ]
        .nunique()
        .gt(0)
        .astype(int)
        .rename(
            "natural_purchase_in_window"
        )
    )

    category_pairs = (
        exposure_frame[
            [
                "exposure_id",
                "golden_customer_id",
                "offer_category",
            ]
        ]
        .merge(
            future[
                [
                    "golden_customer_id",
                    "category",
                ]
            ]
            .drop_duplicates(),
            on="golden_customer_id",
            how="left",
        )
    )

    category_pairs[
        "matched_category"
    ] = (
        category_pairs[
            "offer_category"
        ]
        .eq(
            category_pairs[
                "category"
            ]
        )
    )

    category_response = (
        category_pairs
        .groupby(
            "exposure_id"
        )[
            "matched_category"
        ]
        .any()
        .astype(int)
        .rename(
            "natural_category_purchase_in_window"
        )
    )

    result = exposure_frame[
        [
            "exposure_id",
            "golden_customer_id",
        ]
    ].copy()

    result = result.merge(
        any_purchase,
        left_on="golden_customer_id",
        right_index=True,
        how="left",
    )

    result = result.merge(
        category_response,
        left_on="exposure_id",
        right_index=True,
        how="left",
    )

    result[
        "natural_purchase_in_window"
    ] = (
        result[
            "natural_purchase_in_window"
        ]
        .fillna(0)
        .astype(int)
    )

    result[
        "natural_category_purchase_in_window"
    ] = (
        result[
            "natural_category_purchase_in_window"
        ]
        .fillna(0)
        .astype(int)
    )

    return result[
        [
            "exposure_id",
            "natural_purchase_in_window",
            "natural_category_purchase_in_window",
        ]
    ]


# ---------------------------------------------------------------------
# Campaign generation
# ---------------------------------------------------------------------

def generate_campaign_exposures(
    transactions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    rng = np.random.default_rng(
        RNG_SEED
    )

    all_categories = sorted(
        transactions[
            "category"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if not all_categories:
        raise ValueError(
            "No categories available for campaign generation."
        )

    campaign_rows = []
    truth_rows = []

    exposure_counter = 1

    for campaign_number, campaign_date in enumerate(
        CAMPAIGN_DATES,
        start=1,
    ):

        snapshot = build_customer_snapshot(
            transactions,
            campaign_date,
        )

        if snapshot.empty:
            continue

        preferred_category = (
            build_preferred_category(
                transactions,
                campaign_date,
            )
        )

        # Sample a campaign audience rather than treating every customer.
        audience_share = float(
            rng.uniform(
                0.30,
                0.45,
            )
        )

        audience_size = max(
            1,
            int(
                len(snapshot)
                * audience_share
            ),
        )

        audience = snapshot.sample(
            n=audience_size,
            replace=False,
            random_state=(
                RNG_SEED
                + campaign_number
            ),
        ).copy()

        campaign_id = (
            f"CMP_{campaign_date:%Y%m}_"
            f"{campaign_number:02d}"
        )

        audience[
            "campaign_id"
        ] = campaign_id

        audience[
            "campaign_date"
        ] = campaign_date

        audience[
            "campaign_channel"
        ] = rng.choice(
            CAMPAIGN_CHANNELS,
            size=len(audience),
            replace=True,
            p=[
                0.55,
                0.25,
                0.20,
            ],
        )

        audience[
            "discount_depth"
        ] = rng.choice(
            DISCOUNT_DEPTHS,
            size=len(audience),
            replace=True,
            p=[
                0.25,
                0.35,
                0.25,
                0.15,
            ],
        )

        # Roughly half of offers align to the customer's historically
        # preferred category; the remainder deliberately diversify.
        preferred = (
            audience[
                "golden_customer_id"
            ]
            .map(
                preferred_category
            )
        )

        random_offer = rng.choice(
            all_categories,
            size=len(audience),
            replace=True,
        )

        use_preferred = (
            rng.random(
                len(audience)
            )
            < 0.55
        )

        audience[
            "offer_category"
        ] = np.where(
            use_preferred
            & preferred.notna(),
            preferred,
            random_offer,
        )

        audience[
            "treatment_flag"
        ] = (
            rng.random(
                len(audience)
            )
            < TREATMENT_SHARE
        ).astype(int)

        audience[
            "exposure_id"
        ] = [
            f"EXP_{exposure_counter + i:08d}"
            for i in range(
                len(audience)
            )
        ]

        exposure_counter += len(
            audience
        )

        # ----------------------------------------------------------
        # Synthetic response mechanics
        #
        # This latent response model is hidden ground truth only.
        # The downstream propensity model will not receive these
        # latent components as features.
        # ----------------------------------------------------------

        recency_component = np.clip(
            (
                120
                - audience[
                    "days_since_last_purchase"
                ].to_numpy()
            )
            / 120,
            -1.0,
            1.0,
        )

        frequency_component = np.clip(
            np.log1p(
                audience[
                    "orders_prior_180d"
                ].to_numpy()
            )
            / np.log(
                8
            ),
            0,
            1.5,
        )

        promo_component = np.clip(
            audience[
                "discounted_order_share"
            ]
            .fillna(0)
            .to_numpy(),
            0,
            1,
        )

        value_component = np.clip(
            np.log1p(
                audience[
                    "sales_prior_180d"
                ].to_numpy()
            )
            / np.log(
                2500
            ),
            0,
            1.5,
        )

        category_fit = (
            audience[
                "offer_category"
            ]
            .eq(
                preferred
            )
            .astype(float)
            .to_numpy()
        )

        channel_fit = np.select(
            [
                audience[
                    "campaign_channel"
                ]
                .eq(
                    "Email"
                ),
                audience[
                    "campaign_channel"
                ]
                .eq(
                    "SMS"
                ),
            ],
            [
                0.20,
                0.10,
            ],
            default=0.05,
        )

        discount_effect = (
            (
                audience[
                    "discount_depth"
                ]
                .to_numpy()
                - 0.10
            )
            / 0.15
        )

        month = campaign_date.month

        seasonal_effect = (
            0.20
            if month
            in {
                5,
                8,
                11,
            }
            else 0.0
        )

        baseline_logit = (
            -2.00
            + 0.55
            * recency_component
            + 0.55
            * frequency_component
            + 0.70
            * promo_component
            + 0.35
            * value_component
            + 0.45
            * category_fit
            + channel_fit
            + seasonal_effect
        )

        baseline_probability = sigmoid(
            baseline_logit
        )

        treatment_logit_uplift = (
            0.20
            + 0.55
            * promo_component
            + 0.40
            * category_fit
            + 0.45
            * discount_effect
        )

        treated_probability = sigmoid(
            baseline_logit
            + treatment_logit_uplift
        )

        assigned_probability = np.where(
            audience[
                "treatment_flag"
            ]
            .to_numpy()
            == 1,
            treated_probability,
            baseline_probability,
        )

        # Blend latent conversion with observed natural category purchase
        # behaviour in the transaction history. This keeps the synthetic
        # response layer tied to the existing retail behaviour rather than
        # creating an unrelated marketing universe.
        natural = observed_category_response(
            transactions,
            campaign_date,
            audience,
        )

        audience = audience.merge(
            natural,
            on="exposure_id",
            how="left",
            validate="one_to_one",
        )

        natural_signal = (
            audience[
                "natural_category_purchase_in_window"
            ]
            .fillna(0)
            .to_numpy()
        )

        assigned_probability = np.clip(
            0.85
            * assigned_probability
            + 0.15
            * natural_signal,
            0.01,
            0.95,
        )

        audience[
            "responded"
        ] = (
            rng.random(
                len(audience)
            )
            < assigned_probability
        ).astype(int)

        audience[
            "response_window_days"
        ] = RESPONSE_WINDOW_DAYS

        audience[
            "response_window_end"
        ] = (
            campaign_date
            + pd.Timedelta(
                days=RESPONSE_WINDOW_DAYS
            )
        )

        audience[
            "response_probability_used"
        ] = assigned_probability

        audience[
            "incremental_probability_truth"
        ] = (
            treated_probability
            - baseline_probability
        )

        # Public operational exposure data.
        public_columns = [
            "exposure_id",
            "campaign_id",
            "campaign_date",
            "golden_customer_id",
            "treatment_flag",
            "campaign_channel",
            "offer_category",
            "discount_depth",
            "response_window_days",
            "response_window_end",
            "responded",
        ]

        campaign_rows.append(
            audience[
                public_columns
            ].copy()
        )

        # Hidden QA/uplift truth.
        truth_rows.append(
            audience[
                [
                    "exposure_id",
                    "campaign_id",
                    "campaign_date",
                    "golden_customer_id",
                    "treatment_flag",
                    "natural_purchase_in_window",
                    "natural_category_purchase_in_window",
                    "response_probability_used",
                    "incremental_probability_truth",
                    "responded",
                ]
            ].copy()
        )

    if not campaign_rows:
        raise ValueError(
            "No campaign exposure rows were generated."
        )

    exposures = pd.concat(
        campaign_rows,
        ignore_index=True,
    )

    truth = pd.concat(
        truth_rows,
        ignore_index=True,
    )

    return exposures, truth


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    exposures: pd.DataFrame,
    truth: pd.DataFrame,
) -> None:

    print()
    print("=" * 80)
    print(
        "CAMPAIGN EXPOSURE GENERATION QA"
    )
    print("=" * 80)

    print(
        f"Campaigns                    : "
        f"{exposures['campaign_id'].nunique():,}"
    )

    print(
        f"Exposure rows                : "
        f"{len(exposures):,}"
    )

    print(
        f"Unique Golden Customers      : "
        f"{exposures['golden_customer_id'].nunique():,}"
    )

    print(
        f"Treatment share              : "
        f"{exposures['treatment_flag'].mean():.1%}"
    )

    print(
        f"Overall response rate        : "
        f"{exposures['responded'].mean():.1%}"
    )

    treated = exposures.loc[
        exposures[
            "treatment_flag"
        ]
        .eq(1)
    ]

    control = exposures.loc[
        exposures[
            "treatment_flag"
        ]
        .eq(0)
    ]

    print(
        f"Treated response rate        : "
        f"{treated['responded'].mean():.1%}"
    )

    print(
        f"Control response rate        : "
        f"{control['responded'].mean():.1%}"
    )

    print(
        f"Observed response lift       : "
        f"{treated['responded'].mean() - control['responded'].mean():+.1%}"
    )

    print(
        f"Median discount depth        : "
        f"{exposures['discount_depth'].median():.0%}"
    )

    print()
    print(
        "Response rate by campaign:"
    )

    campaign_summary = (
        exposures
        .groupby(
            [
                "campaign_date",
                "treatment_flag",
            ]
        )
        .agg(
            exposures=(
                "exposure_id",
                "size",
            ),
            response_rate=(
                "responded",
                "mean",
            ),
        )
        .reset_index()
    )

    campaign_summary[
        "response_rate"
    ] = (
        campaign_summary[
            "response_rate"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )

    print(
        campaign_summary
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Campaign channel mix:"
    )

    print(
        exposures[
            "campaign_channel"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Offer category mix:"
    )

    print(
        exposures[
            "offer_category"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Hidden ground-truth fields are written only to "
        "campaign_exposure_ground_truth.parquet and must "
        "not be used as propensity-model features."
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
        "Generating synthetic campaign exposures..."
    )

    exposures, truth = (
        generate_campaign_exposures(
            transactions
        )
    )

    exposures.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    truth.to_parquet(
        GROUND_TRUTH_FILE,
        index=False,
    )

    run_qa(
        exposures,
        truth,
    )

    print()
    print(
        "Files created:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        GROUND_TRUTH_FILE
    )


if __name__ == "__main__":
    main()
