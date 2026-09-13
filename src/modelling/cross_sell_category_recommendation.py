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

PROPENSITY_SCORE_FILE = Path(
    "data/runtime/cross_sell_model_scores.parquet"
)

OUTPUT_FILE = Path(
    "data/runtime/cross_sell_category_recommendations.parquet"
)

VALIDATION_FILE = Path(
    "data/runtime/cross_sell_category_validation.parquet"
)

CANDIDATE_AUDIT_FILE = Path(
    "data/runtime/cross_sell_category_candidate_audit.parquet"
)

OUTCOME_DAYS = 180

WEIGHT_ADJACENCY = 0.50
WEIGHT_PENETRATION = 0.25
WEIGHT_MARGIN = 0.25


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def min_max_scale(
    series: pd.Series,
) -> pd.Series:

    values = series.astype(float)

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            0.0,
            index=series.index,
            dtype=float,
        )

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            0.5,
            index=series.index,
            dtype=float,
        )

    return (
        values
        - minimum
    ) / (
        maximum
        - minimum
    )


def prepare_transactions(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    tx = transactions.copy()

    required = {
        "golden_customer_id",
        "transaction_date",
        "order_id",
        "category",
        "net_sales",
        "gross_margin",
    }

    missing = required - set(
        tx.columns
    )

    if missing:
        raise ValueError(
            "Golden customer transactions are missing "
            f"required columns: {sorted(missing)}"
        )

    tx[
        "transaction_date"
    ] = pd.to_datetime(
        tx[
            "transaction_date"
        ]
    )

    return tx


# ---------------------------------------------------------------------
# Historical category evidence
# ---------------------------------------------------------------------

def build_customer_category_sets(
    history: pd.DataFrame,
) -> dict[str, set[str]]:

    return (
        history
        .groupby(
            "golden_customer_id"
        )[
            "category"
        ]
        .agg(
            lambda s:
                set(
                    s.dropna()
                )
        )
        .to_dict()
    )


def build_category_penetration(
    history: pd.DataFrame,
) -> pd.DataFrame:

    customer_category = (
        history[
            [
                "golden_customer_id",
                "category",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    customer_count = (
        customer_category[
            "golden_customer_id"
        ]
        .nunique()
    )

    penetration = (
        customer_category
        .groupby(
            "category"
        )
        .agg(
            customers=(
                "golden_customer_id",
                "nunique",
            )
        )
        .reset_index()
    )

    penetration[
        "category_penetration"
    ] = np.where(
        customer_count > 0,
        penetration[
            "customers"
        ] / customer_count,
        0.0,
    )

    return penetration[
        [
            "category",
            "category_penetration",
        ]
    ]


def build_category_margin_profile(
    history: pd.DataFrame,
) -> pd.DataFrame:

    category_margin = (
        history
        .groupby(
            "category"
        )
        .agg(
            category_orders=(
                "order_id",
                "nunique",
            ),
            category_sales=(
                "net_sales",
                "sum",
            ),
            category_margin=(
                "gross_margin",
                "sum",
            ),
        )
        .reset_index()
    )

    category_margin[
        "margin_per_order"
    ] = np.where(
        category_margin[
            "category_orders"
        ] > 0,
        category_margin[
            "category_margin"
        ] / category_margin[
            "category_orders"
        ],
        0.0,
    )

    category_margin[
        "margin_rate"
    ] = np.where(
        category_margin[
            "category_sales"
        ] > 0,
        category_margin[
            "category_margin"
        ] / category_margin[
            "category_sales"
        ],
        0.0,
    )

    category_margin[
        "margin_score"
    ] = min_max_scale(
        category_margin[
            "margin_per_order"
        ]
    )

    return category_margin[
        [
            "category",
            "margin_per_order",
            "margin_rate",
            "margin_score",
        ]
    ]


def build_category_adjacency(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build category affinity using customer-level co-occurrence.

    For source category A and candidate category B:
        adjacency = customers buying both A and B
                    / customers buying A

    This is directional and uses history only.
    """

    customer_category = (
        history[
            [
                "golden_customer_id",
                "category",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    source_counts = (
        customer_category
        .groupby(
            "category"
        )
        .agg(
            source_customers=(
                "golden_customer_id",
                "nunique",
            )
        )
        .reset_index()
        .rename(
            columns={
                "category":
                    "source_category"
            }
        )
    )

    pairs = (
        customer_category
        .merge(
            customer_category,
            on="golden_customer_id",
            suffixes=(
                "_source",
                "_candidate",
            ),
        )
    )

    pairs = pairs.loc[
        pairs[
            "category_source"
        ]
        .ne(
            pairs[
                "category_candidate"
            ]
        )
    ].copy()

    pair_counts = (
        pairs
        .groupby(
            [
                "category_source",
                "category_candidate",
            ]
        )
        .agg(
            shared_customers=(
                "golden_customer_id",
                "nunique",
            )
        )
        .reset_index()
        .rename(
            columns={
                "category_source":
                    "source_category",
                "category_candidate":
                    "candidate_category",
            }
        )
    )

    adjacency = (
        pair_counts
        .merge(
            source_counts,
            on="source_category",
            how="left",
            validate="many_to_one",
        )
    )

    adjacency[
        "adjacency_rate"
    ] = np.where(
        adjacency[
            "source_customers"
        ] > 0,
        adjacency[
            "shared_customers"
        ] / adjacency[
            "source_customers"
        ],
        0.0,
    )

    return adjacency[
        [
            "source_category",
            "candidate_category",
            "adjacency_rate",
        ]
    ]


# ---------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------

def score_customer_categories(
    customer_id: str,
    purchased_categories: set[str],
    all_categories: list[str],
    adjacency: pd.DataFrame,
    penetration: pd.DataFrame,
    margin_profile: pd.DataFrame,
) -> pd.DataFrame:

    candidates = [
        category
        for category in all_categories
        if category
        not in purchased_categories
    ]

    if not candidates:
        return pd.DataFrame()

    candidate_df = pd.DataFrame(
        {
            "golden_customer_id":
                customer_id,
            "candidate_category":
                candidates,
        }
    )

    if purchased_categories:
        customer_adjacency = (
            adjacency.loc[
                adjacency[
                    "source_category"
                ]
                .isin(
                    purchased_categories
                )
                & adjacency[
                    "candidate_category"
                ]
                .isin(
                    candidates
                )
            ]
            .groupby(
                "candidate_category"
            )
            .agg(
                adjacency_score=(
                    "adjacency_rate",
                    "mean",
                ),
                strongest_adjacency=(
                    "adjacency_rate",
                    "max",
                ),
            )
            .reset_index()
        )
    else:
        customer_adjacency = pd.DataFrame(
            columns=[
                "candidate_category",
                "adjacency_score",
                "strongest_adjacency",
            ]
        )

    candidate_df = (
        candidate_df
        .merge(
            customer_adjacency,
            on="candidate_category",
            how="left",
            validate="one_to_one",
        )
        .merge(
            penetration.rename(
                columns={
                    "category":
                        "candidate_category"
                }
            ),
            on="candidate_category",
            how="left",
            validate="one_to_one",
        )
        .merge(
            margin_profile.rename(
                columns={
                    "category":
                        "candidate_category"
                }
            ),
            on="candidate_category",
            how="left",
            validate="one_to_one",
        )
    )

    for column in [
        "adjacency_score",
        "strongest_adjacency",
        "category_penetration",
        "margin_score",
        "margin_per_order",
        "margin_rate",
    ]:
        candidate_df[
            column
        ] = (
            candidate_df[
                column
            ]
            .fillna(0.0)
        )

    candidate_df[
        "category_affinity_score"
    ] = 100 * (
        WEIGHT_ADJACENCY
        * candidate_df[
            "adjacency_score"
        ]
        + WEIGHT_PENETRATION
        * candidate_df[
            "category_penetration"
        ]
        + WEIGHT_MARGIN
        * candidate_df[
            "margin_score"
        ]
    )

    candidate_df = (
        candidate_df
        .sort_values(
            [
                "category_affinity_score",
                "adjacency_score",
                "margin_per_order",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    candidate_df[
        "category_rank"
    ] = (
        candidate_df.index
        + 1
    )

    return candidate_df


# ---------------------------------------------------------------------
# OOT recommendation build
# ---------------------------------------------------------------------

def build_recommendations(
    transactions: pd.DataFrame,
    propensity_scores: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if propensity_scores.empty:
        raise ValueError(
            "Cross-sell propensity score file is empty."
        )

    if propensity_scores[
        "model_status"
    ].nunique() != 1:
        raise ValueError(
            "Expected one governed model status "
            "in cross-sell score output."
        )

    model_status = str(
        propensity_scores[
            "model_status"
        ].iloc[0]
    )

    if model_status != "ACCEPTED":
        raise ValueError(
            "Cross-sell category recommendation cannot "
            "be activated because the propensity model "
            f"status is {model_status}."
        )

    observation_dates = (
        pd.to_datetime(
            propensity_scores[
                "observation_date"
            ]
        )
        .dropna()
        .unique()
    )

    if len(
        observation_dates
    ) != 1:
        raise ValueError(
            "Category recommendation validation expects "
            "one OOT observation date."
        )

    observation_date = pd.Timestamp(
        observation_dates[0]
    )

    history = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .le(
            observation_date
        )
    ].copy()

    all_categories = sorted(
        transactions[
            "category"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    customer_category_sets = (
        build_customer_category_sets(
            history
        )
    )

    adjacency = (
        build_category_adjacency(
            history
        )
    )

    penetration = (
        build_category_penetration(
            history
        )
    )

    margin_profile = (
        build_category_margin_profile(
            history
        )
    )

    recommendation_rows = []
    candidate_rows = []

    for row in propensity_scores.itertuples(
        index=False
    ):

        customer_id = str(
            row.golden_customer_id
        )

        purchased_categories = (
            customer_category_sets.get(
                customer_id,
                set(),
            )
        )

        scored = (
            score_customer_categories(
                customer_id,
                purchased_categories,
                all_categories,
                adjacency,
                penetration,
                margin_profile,
            )
        )

        if scored.empty:
            continue

        scored[
            "observation_date"
        ] = observation_date

        scored[
            "cross_sell_propensity_180d"
        ] = float(
            row.cross_sell_propensity_180d
        )

        candidate_rows.append(
            scored
        )

        top = scored.iloc[0]

        second = (
            scored.iloc[1]
            if len(scored) > 1
            else None
        )

        third = (
            scored.iloc[2]
            if len(scored) > 2
            else None
        )

        score_gap = (
            float(
                top[
                    "category_affinity_score"
                ]
            )
            - float(
                second[
                    "category_affinity_score"
                ]
            )
            if second is not None
            else float(
                top[
                    "category_affinity_score"
                ]
            )
        )

        recommendation_confidence = (
            "High"
            if score_gap >= 10
            else (
                "Medium"
                if score_gap >= 5
                else "Low"
            )
        )

        recommendation_rows.append(
            {
                "golden_customer_id":
                    customer_id,
                "observation_date":
                    observation_date,
                "cross_sell_propensity_180d":
                    float(
                        row.cross_sell_propensity_180d
                    ),
                "cross_sell_propensity_band":
                    getattr(
                        row,
                        "cross_sell_propensity_band",
                        None,
                    ),
                "recommended_category":
                    top[
                        "candidate_category"
                    ],
                "category_affinity_score":
                    float(
                        top[
                            "category_affinity_score"
                        ]
                    ),
                "adjacency_score":
                    float(
                        top[
                            "adjacency_score"
                        ]
                    ),
                "category_penetration":
                    float(
                        top[
                            "category_penetration"
                        ]
                    ),
                "category_margin_per_order":
                    float(
                        top[
                            "margin_per_order"
                        ]
                    ),
                "second_best_category":
                    (
                        second[
                            "candidate_category"
                        ]
                        if second is not None
                        else None
                    ),
                "third_best_category":
                    (
                        third[
                            "candidate_category"
                        ]
                        if third is not None
                        else None
                    ),
                "category_score_gap":
                    score_gap,
                "recommendation_confidence":
                    recommendation_confidence,
                "recommendation_reason":
                    (
                        "Highest category opportunity based on "
                        "historical category adjacency, category "
                        "penetration and margin economics."
                    ),
                "model_status":
                    model_status,
            }
        )

    recommendations = pd.DataFrame(
        recommendation_rows
    )

    candidate_audit = (
        pd.concat(
            candidate_rows,
            ignore_index=True,
        )
        if candidate_rows
        else pd.DataFrame()
    )

    return (
        recommendations,
        candidate_audit,
    )


# ---------------------------------------------------------------------
# Historical validation
# ---------------------------------------------------------------------

def validate_recommendations(
    recommendations: pd.DataFrame,
    candidate_audit: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    if recommendations.empty:
        return pd.DataFrame()

    observation_date = pd.Timestamp(
        recommendations[
            "observation_date"
        ]
        .iloc[0]
    )

    outcome_end = (
        observation_date
        + pd.Timedelta(
            days=OUTCOME_DAYS
        )
    )

    history = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .le(
            observation_date
        )
    ]

    future = transactions.loc[
        transactions[
            "transaction_date"
        ]
        .gt(
            observation_date
        )
        & transactions[
            "transaction_date"
        ]
        .le(
            outcome_end
        )
    ]

    prior_sets = (
        build_customer_category_sets(
            history
        )
    )

    future_sets = (
        build_customer_category_sets(
            future
        )
    )

    validation_rows = []

    for row in recommendations.itertuples(
        index=False
    ):

        customer_id = str(
            row.golden_customer_id
        )

        prior = prior_sets.get(
            customer_id,
            set(),
        )

        future_categories = (
            future_sets.get(
                customer_id,
                set(),
            )
        )

        actual_new_categories = (
            future_categories
            - prior
        )

        candidates = (
            candidate_audit.loc[
                candidate_audit[
                    "golden_customer_id"
                ]
                .eq(
                    customer_id
                )
            ]
            .sort_values(
                "category_rank"
            )
        )

        ranked_categories = (
            candidates[
                "candidate_category"
            ]
            .tolist()
        )

        top_1 = set(
            ranked_categories[:1]
        )

        top_2 = set(
            ranked_categories[:2]
        )

        top_3 = set(
            ranked_categories[:3]
        )

        validation_rows.append(
            {
                "golden_customer_id":
                    customer_id,
                "actual_cross_sell":
                    int(
                        len(
                            actual_new_categories
                        )
                        > 0
                    ),
                "actual_new_categories":
                    " | ".join(
                        sorted(
                            actual_new_categories
                        )
                    ),
                "top1_hit":
                    int(
                        bool(
                            actual_new_categories
                            & top_1
                        )
                    ),
                "top2_hit":
                    int(
                        bool(
                            actual_new_categories
                            & top_2
                        )
                    ),
                "top3_hit":
                    int(
                        bool(
                            actual_new_categories
                            & top_3
                        )
                    ),
                "recommended_category":
                    row.recommended_category,
                "second_best_category":
                    row.second_best_category,
                "third_best_category":
                    row.third_best_category,
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    return validation


# ---------------------------------------------------------------------
# QA reporting
# ---------------------------------------------------------------------

def run_qa(
    recommendations: pd.DataFrame,
    validation: pd.DataFrame,
    candidate_audit: pd.DataFrame,
) -> None:

    print()
    print("=" * 86)
    print("CROSS-SELL CATEGORY RECOMMENDATION QA")
    print("=" * 86)

    print(
        f"Golden customers recommended      : "
        f"{len(recommendations):,}"
    )

    print(
        f"Unique Golden Customer IDs        : "
        f"{recommendations['golden_customer_id'].nunique():,}"
    )

    print(
        f"Candidate category rows           : "
        f"{len(candidate_audit):,}"
    )

    print(
        f"Recommended categories            : "
        f"{recommendations['recommended_category'].nunique():,}"
    )

    positive = validation.loc[
        validation[
            "actual_cross_sell"
        ]
        .eq(1)
    ].copy()

    print(
        f"OOT customers entering new category: "
        f"{len(positive):,}"
    )

    if not positive.empty:
        print(
            f"Top-1 category hit rate           : "
            f"{positive['top1_hit'].mean():.1%}"
        )

        print(
            f"Top-2 category hit rate           : "
            f"{positive['top2_hit'].mean():.1%}"
        )

        print(
            f"Top-3 category hit rate           : "
            f"{positive['top3_hit'].mean():.1%}"
        )

    print()
    print(
        "Recommendation confidence:"
    )

    print(
        recommendations[
            "recommendation_confidence"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Recommended category mix:"
    )

    print(
        recommendations[
            "recommended_category"
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
        "Loading accepted cross-sell propensity scores..."
    )

    propensity_scores = pd.read_parquet(
        PROPENSITY_SCORE_FILE
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
        "Building category recommendations..."
    )

    (
        recommendations,
        candidate_audit,
    ) = build_recommendations(
        transactions,
        propensity_scores,
    )

    if recommendations.empty:
        raise ValueError(
            "No cross-sell category recommendations "
            "were created."
        )

    validation = validate_recommendations(
        recommendations,
        candidate_audit,
        transactions,
    )

    recommendations.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    validation.to_parquet(
        VALIDATION_FILE,
        index=False,
    )

    candidate_audit.to_parquet(
        CANDIDATE_AUDIT_FILE,
        index=False,
    )

    run_qa(
        recommendations,
        validation,
        candidate_audit,
    )

    print()
    print(
        "Files created:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        VALIDATION_FILE
    )

    print(
        CANDIDATE_AUDIT_FILE
    )


if __name__ == "__main__":
    main()
