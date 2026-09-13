from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PRIORITY_FILE = Path("data/runtime/customer_priority.parquet")
LTV_FILE = Path("data/runtime/customer_ltv.parquet")
CLUSTER_FILE = Path("data/runtime/customer_clusters.parquet")
TRANSACTION_FILE = Path("data/runtime/golden_customer_transactions.parquet")
OUTPUT_FILE = Path("data/runtime/customer_next_best_action.parquet")
CROSS_SELL_SCORE_FILE = Path("data/runtime/cross_sell_model_scores.parquet")
CROSS_SELL_RECOMMENDATION_FILE = Path("data/runtime/cross_sell_category_recommendations.parquet")
CROSS_SELL_GOVERNANCE_FILE = Path("data/runtime/cross_sell_model_governance.parquet")
PROMOTION_SCORE_FILE = Path("data/runtime/promotion_response_operational_scores.parquet")
PROMOTION_GOVERNANCE_FILE = Path("data/runtime/promotion_response_model_governance.parquet")

CANDIDATE_ACTIONS = [
    "Protect",
    "Re-engage",
    "Develop",
    "Cross-sell",
    "Promote",
    "Do Nothing",
]

ACTION_COST = {
    "Protect": 18.0,
    "Re-engage": 12.0,
    "Develop": 8.0,
    "Cross-sell": 6.0,
    "Promote": 10.0,
    "Do Nothing": 0.0,
}

ACTION_RESPONSE_BASE = {
    "Protect": 0.30,
    "Re-engage": 0.24,
    "Develop": 0.22,
    "Cross-sell": 0.18,
    "Promote": 0.20,
    "Do Nothing": 0.00,
}

ACTION_SALES_UPLIFT = {
    "Protect": 0.18,
    "Re-engage": 0.16,
    "Develop": 0.14,
    "Cross-sell": 0.12,
    "Promote": 0.10,
    "Do Nothing": 0.00,
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def score_to_probability(score: pd.Series) -> pd.Series:
    return (score.fillna(0) / 100).clip(0, 1)


# ---------------------------------------------------------------------
# Transaction-derived behaviour
# ---------------------------------------------------------------------

def build_transaction_behaviour(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    tx["discounted_order"] = tx["discount_pct"].fillna(0).gt(0)

    base = (
        tx.groupby("golden_customer_id")
        .agg(
            observed_orders_tx=("order_id", "nunique"),
            observed_sales_tx=("net_sales", "sum"),
            observed_margin_tx=("gross_margin", "sum"),
            average_order_value_tx=("net_sales", "mean"),
            average_discount_pct=("discount_pct", "mean"),
            discounted_order_share=("discounted_order", "mean"),
            categories_used_tx=("category", "nunique"),
            channels_used_tx=("channel", "nunique"),
        )
        .reset_index()
    )

    base["observed_margin_rate_tx"] = safe_divide(
        base["observed_margin_tx"],
        base["observed_sales_tx"],
    )

    category_counts = (
        tx.groupby(["golden_customer_id", "category"])
        .agg(
            category_orders=("order_id", "nunique"),
            category_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    category_counts["total_category_orders"] = (
        category_counts.groupby("golden_customer_id")["category_orders"].transform("sum")
    )
    category_counts["category_order_share"] = safe_divide(
        category_counts["category_orders"],
        category_counts["total_category_orders"],
    )

    dominant_category = (
        category_counts.sort_values(
            ["golden_customer_id", "category_order_share", "category_margin"],
            ascending=[True, False, False],
        )
        .drop_duplicates("golden_customer_id")
        [["golden_customer_id", "category", "category_order_share"]]
        .rename(
            columns={
                "category": "dominant_category",
                "category_order_share": "dominant_category_share",
            }
        )
    )

    channel_counts = (
        tx.groupby(["golden_customer_id", "channel"])
        .agg(channel_orders=("order_id", "nunique"))
        .reset_index()
    )
    channel_counts["total_channel_orders"] = (
        channel_counts.groupby("golden_customer_id")["channel_orders"].transform("sum")
    )
    channel_counts["channel_order_share"] = safe_divide(
        channel_counts["channel_orders"],
        channel_counts["total_channel_orders"],
    )

    preferred_channel = (
        channel_counts.sort_values(
            ["golden_customer_id", "channel_order_share"],
            ascending=[True, False],
        )
        .drop_duplicates("golden_customer_id")
        [["golden_customer_id", "channel", "channel_order_share"]]
        .rename(
            columns={
                "channel": "preferred_channel",
                "channel_order_share": "preferred_channel_share",
            }
        )
    )

    return (
        base.merge(dominant_category, on="golden_customer_id", how="left", validate="one_to_one")
        .merge(preferred_channel, on="golden_customer_id", how="left", validate="one_to_one")
    )


# ---------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------

def build_nba_features(
    priority: pd.DataFrame,
    customer_ltv: pd.DataFrame,
    clusters: pd.DataFrame,
    transaction_behaviour: pd.DataFrame,
    cross_sell_scores: pd.DataFrame | None = None,
    cross_sell_recommendations: pd.DataFrame | None = None,
    cross_sell_model_accepted: bool = False,
    promotion_scores: pd.DataFrame | None = None,
    promotion_model_accepted: bool = False,
) -> pd.DataFrame:
    result = priority.copy()

    ltv_cols = [
        "golden_customer_id",
        "observed_ltv_sales",
        "observed_ltv_margin",
        "observed_margin_rate",
        "repeat_customer",
        "days_to_second_purchase",
        "m6_margin",
        "m12_margin",
        "m12_mature",
        "m24_margin",
        "m24_mature",
        "purchase_tenure_days",
    ]
    ltv_cols = [c for c in ltv_cols if c in customer_ltv.columns]
    result = result.merge(
        customer_ltv[ltv_cols],
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    cluster_cols = [c for c in ["golden_customer_id", "cluster_name"] if c in clusters.columns]
    result = result.merge(
        clusters[cluster_cols].drop_duplicates("golden_customer_id"),
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        transaction_behaviour,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    result["cluster_name"] = result["cluster_name"].fillna("Not Yet Clustered")
    result["observed_ltv_sales"] = result["observed_ltv_sales"].fillna(0)
    result["observed_ltv_margin"] = result["observed_ltv_margin"].fillna(0)

    fallback_margin_rate = safe_divide(
        result["trailing_12m_margin"].fillna(0),
        result["trailing_12m_sales"].fillna(0),
    )
    result["observed_margin_rate"] = (
        result["observed_margin_rate"]
        .fillna(fallback_margin_rate)
        .fillna(0.30)
        .clip(0.10, 0.60)
    )

    result["discounted_order_share"] = result["discounted_order_share"].fillna(0).clip(0, 1)
    result["dominant_category_share"] = result["dominant_category_share"].fillna(0).clip(0, 1)
    result["preferred_channel_share"] = result["preferred_channel_share"].fillna(0).clip(0, 1)
    result["average_discount_pct"] = result["average_discount_pct"].fillna(0)
    result["categories_used_tx"] = result["categories_used_tx"].fillna(0)

    result["category_headroom_score"] = (1 - result["dominant_category_share"]).clip(0, 1)
    result["promo_responsiveness_score"] = (
        0.70 * result["discounted_order_share"]
        + 0.30 * (result["average_discount_pct"] / 0.25).clip(0, 1)
    ).clip(0, 1)

    result["lapse_probability_proxy"] = score_to_probability(result["lapse_risk_score"])
    result["momentum_probability_proxy"] = score_to_probability(result["momentum_risk_score"])
    result["value_probability_proxy"] = score_to_probability(result["value_score"])
    result["risk_intensity"] = (
        0.60 * result["lapse_probability_proxy"]
        + 0.40 * result["momentum_probability_proxy"]
    ).clip(0, 1)

    result["expected_baseline_sales"] = result["trailing_12m_sales"].fillna(0)
    low_recent = result["expected_baseline_sales"].le(0)
    result.loc[low_recent, "expected_baseline_sales"] = (
        result.loc[low_recent, "average_order_value_tx"].fillna(0)
    )

    # Governed cross-sell model evidence. The model is advisory only when
    # its governance artefact says ACCEPTED; otherwise the existing
    # transparent rule-based cross-sell pathway remains in force.
    result["cross_sell_model_accepted"] = bool(cross_sell_model_accepted)
    result["cross_sell_propensity"] = np.nan
    result["cross_sell_propensity_source"] = "Rule-based fallback"
    result["recommended_cross_sell_category"] = pd.NA
    result["cross_sell_category_confidence"] = pd.NA
    result["cross_sell_category_affinity"] = np.nan

    if cross_sell_scores is not None and not cross_sell_scores.empty:
        scores = cross_sell_scores.copy()
        id_col = "golden_customer_id"
        probability_candidates = [
            "cross_sell_propensity_180d",
            "calibrated_probability",
            "cross_sell_propensity",
            "predicted_probability",
            "probability",
        ]
        probability_col = next(
            (c for c in probability_candidates if c in scores.columns),
            None,
        )
        if id_col in scores.columns and probability_col is not None:
            score_subset = (
                scores[[id_col, probability_col]]
                .dropna(subset=[id_col])
                .drop_duplicates(id_col, keep="last")
                .rename(columns={probability_col: "_cross_sell_model_probability"})
            )
            result = result.merge(
                score_subset,
                on=id_col,
                how="left",
                validate="one_to_one",
            )
            if cross_sell_model_accepted:
                result["cross_sell_propensity"] = (
                    pd.to_numeric(
                        result["_cross_sell_model_probability"],
                        errors="coerce",
                    )
                    .clip(0, 1)
                )
                result.loc[
                    result["cross_sell_propensity"].notna(),
                    "cross_sell_propensity_source",
                ] = "Accepted governed model"
            result = result.drop(
                columns=["_cross_sell_model_probability"],
                errors="ignore",
            )

    if cross_sell_recommendations is not None and not cross_sell_recommendations.empty:
        recs = cross_sell_recommendations.copy()
        rename_map = {}
        for candidate in ["recommended_category", "top_category", "category"]:
            if candidate in recs.columns:
                rename_map[candidate] = "recommended_cross_sell_category"
                break
        for candidate in [
            "recommendation_confidence",
            "category_confidence",
            "confidence",
        ]:
            if candidate in recs.columns:
                rename_map[candidate] = "cross_sell_category_confidence"
                break
        for candidate in [
            "affinity_score",
            "category_affinity",
            "recommendation_score",
        ]:
            if candidate in recs.columns:
                rename_map[candidate] = "cross_sell_category_affinity"
                break

        keep = ["golden_customer_id"] + list(rename_map)
        if "golden_customer_id" in recs.columns and rename_map:
            rec_subset = (
                recs[keep]
                .rename(columns=rename_map)
                .drop_duplicates("golden_customer_id", keep="last")
            )
            # Avoid collisions with the placeholder columns already created.
            for col in [
                "recommended_cross_sell_category",
                "cross_sell_category_confidence",
                "cross_sell_category_affinity",
            ]:
                if col in rec_subset.columns:
                    mapper = rec_subset.set_index("golden_customer_id")[col]
                    result[col] = result["golden_customer_id"].map(mapper)

    # Governed promotion-response model evidence. Only an ACCEPTED model
    # may replace the transparent promotion-response fallback used by NBA.
    result["promotion_model_accepted"] = bool(promotion_model_accepted)
    result["promotion_response_propensity"] = np.nan
    result["promotion_propensity_source"] = "Decision-engine fallback"
    result["recommended_campaign_channel"] = pd.NA
    result["recommended_discount_depth"] = np.nan
    result["recommended_offer_category"] = pd.NA
    result["promotion_response_band"] = pd.NA

    if promotion_scores is not None and not promotion_scores.empty:
        scores = promotion_scores.copy()
        id_col = "golden_customer_id"

        probability_candidates = [
            "promotion_response_propensity",
            "response_propensity",
            "calibrated_probability",
            "predicted_probability",
            "probability",
        ]
        probability_col = next(
            (c for c in probability_candidates if c in scores.columns),
            None,
        )

        rename_candidates = {
            "campaign_channel": "recommended_campaign_channel",
            "recommended_campaign_channel": "recommended_campaign_channel",
            "discount_depth": "recommended_discount_depth",
            "recommended_discount_depth": "recommended_discount_depth",
            "offer_category": "recommended_offer_category",
            "recommended_offer_category": "recommended_offer_category",
            "promotion_response_band": "promotion_response_band",
            "response_band": "promotion_response_band",
        }

        if id_col in scores.columns:
            keep = [id_col]
            rename_map = {}

            if probability_col is not None:
                keep.append(probability_col)
                rename_map[probability_col] = "_promotion_model_probability"

            for source_col, target_col in rename_candidates.items():
                if source_col in scores.columns and source_col not in keep:
                    # Keep the first available source for each target.
                    if target_col not in rename_map.values():
                        keep.append(source_col)
                        rename_map[source_col] = target_col

            score_subset = (
                scores[keep]
                .dropna(subset=[id_col])
                .drop_duplicates(id_col, keep="last")
                .rename(columns=rename_map)
            )

            # Map rather than merge so placeholder columns remain stable.
            score_subset = score_subset.set_index(id_col)

            if "_promotion_model_probability" in score_subset.columns:
                mapped = result[id_col].map(
                    score_subset["_promotion_model_probability"]
                )
                if promotion_model_accepted:
                    result["promotion_response_propensity"] = (
                        pd.to_numeric(mapped, errors="coerce")
                        .clip(0, 1)
                    )
                    result.loc[
                        result["promotion_response_propensity"].notna(),
                        "promotion_propensity_source",
                    ] = "Accepted governed model"

            for col in [
                "recommended_campaign_channel",
                "recommended_discount_depth",
                "recommended_offer_category",
                "promotion_response_band",
            ]:
                if col in score_subset.columns:
                    result[col] = result[id_col].map(score_subset[col])

    return result


# ---------------------------------------------------------------------
# Candidate action scores
# ---------------------------------------------------------------------

def add_action_scores(customers: pd.DataFrame) -> pd.DataFrame:
    result = customers.copy()

    high_value = (
        result["customer_value_tier"]
        .isin(["High", "Very High"])
    )
    meaningful_value = (
        result["customer_value_tier"]
        .isin(["Medium", "High", "Very High"])
    )
    active = (
        result["active_customer"]
        .fillna(False)
    )
    no_purchase = (
        result["rfm_segment"]
        .eq("No Purchase")
    )
    developing = (
        result["rfm_segment"]
        .eq("Developing")
    )
    repeat = (
        result["repeat_customer"]
        .fillna(False)
    )

    hard_lapse = (
        result["lifecycle_status"]
        .isin(["At Risk", "Highly Lapsed"])
    )
    watch = (
        result["lifecycle_status"]
        .eq("Watch")
    )
    deteriorating = (
        result["cadence_trend_status"]
        .isin(["Deteriorating", "Strongly Deteriorating"])
    )
    strong_deterioration = (
        result["cadence_trend_status"]
        .eq("Strongly Deteriorating")
    )

    low_risk = (
        result["risk_intensity"] < 0.45
    )
    moderate_or_lower_risk = (
        result["risk_intensity"] < 0.60
    )

    category_whitespace = (
        result["category_headroom_score"] >= 0.35
    )
    credible_category_history = (
        result["categories_used_tx"]
        .fillna(0)
        .between(2, 4)
    )
    promo_responsive = (
        result["promo_responsiveness_score"] >= 0.55
    )

    # --------------------------------------------------------------
    # Commercial eligibility gates
    #
    # Protect and Re-engage take precedence over growth actions.
    # Cross-sell and Promote require healthy enough customers plus
    # evidence that the specific treatment is relevant.
    # --------------------------------------------------------------

    result["eligible_protect"] = (
        high_value
        & (
            hard_lapse
            | strong_deterioration
            | (
                deteriorating
                & (result["risk_intensity"] >= 0.55)
            )
        )
    )

    result["eligible_reengage"] = (
        meaningful_value
        & repeat
        & (
            hard_lapse
            | (
                ~active
                & (result["risk_intensity"] >= 0.45)
            )
        )
        & ~result["eligible_protect"]
    )

    result["eligible_develop"] = (
        active
        & moderate_or_lower_risk
        & (
            developing
            | (
                result["customer_value_tier"]
                .isin(["Low", "Medium"])
                & repeat
            )
        )
    )

    result["eligible_cross_sell"] = (
        active
        & low_risk
        & meaningful_value
        & repeat
        & category_whitespace
        & credible_category_history
    )

    result["eligible_promote"] = (
        active
        & low_risk
        & meaningful_value
        & promo_responsive
    )

    result["eligible_do_nothing"] = True

    # --------------------------------------------------------------
    # Candidate relevance scores
    # --------------------------------------------------------------

    result["score_protect"] = 100 * (
        0.30 * result["value_probability_proxy"]
        + 0.40 * result["risk_intensity"]
        + 0.20 * high_value.astype(float)
        + 0.10 * deteriorating.astype(float)
    )

    result["score_reengage"] = 100 * (
        0.45 * result["risk_intensity"]
        + 0.20 * meaningful_value.astype(float)
        + 0.20 * (~active).astype(float)
        + 0.15 * repeat.astype(float)
    )

    result["score_develop"] = 100 * (
        0.35 * developing.astype(float)
        + 0.25 * active.astype(float)
        + 0.20 * result["value_probability_proxy"]
        + 0.20 * (1 - result["risk_intensity"])
    )

    rule_cross_sell_score = (
        0.20 * active.astype(float)
        + 0.20 * meaningful_value.astype(float)
        + 0.30 * result["category_headroom_score"]
        + 0.15 * repeat.astype(float)
        + 0.15 * (1 - result["dominant_category_share"])
    )

    governed_propensity = result["cross_sell_propensity"].fillna(rule_cross_sell_score)
    use_governed_propensity = (
        result["cross_sell_model_accepted"].fillna(False)
        & result["cross_sell_propensity"].notna()
    )

    result["score_cross_sell"] = 100 * np.where(
        use_governed_propensity,
        0.60 * governed_propensity + 0.40 * rule_cross_sell_score,
        rule_cross_sell_score,
    )

    rule_promote_score = (
        0.50 * result["promo_responsiveness_score"]
        + 0.15 * active.astype(float)
        + 0.15 * (1 - result["risk_intensity"])
        + 0.20 * meaningful_value.astype(float)
    )

    use_governed_promotion = (
        result["promotion_model_accepted"].fillna(False)
        & result["promotion_response_propensity"].notna()
    )

    result["score_promote"] = 100 * np.where(
        use_governed_promotion,
        0.60 * result["promotion_response_propensity"]
        + 0.40 * rule_promote_score,
        rule_promote_score,
    )

    result["score_do_nothing"] = 100 * (
        0.50 * (1 - result["risk_intensity"])
        + 0.25 * (1 - result["value_probability_proxy"])
        + 0.15 * no_purchase.astype(float)
        + 0.10 * (
            ~(
                result["eligible_protect"]
                | result["eligible_reengage"]
                | result["eligible_develop"]
                | result["eligible_cross_sell"]
                | result["eligible_promote"]
            )
        ).astype(float)
    )

    for column in [
        "score_protect",
        "score_reengage",
        "score_develop",
        "score_cross_sell",
        "score_promote",
        "score_do_nothing",
    ]:
        result[column] = (
            result[column]
            .fillna(0)
            .clip(0, 100)
            .round(1)
        )

    return result


# ---------------------------------------------------------------------
# Intervention economics
# ---------------------------------------------------------------------

def calculate_action_economics(customers: pd.DataFrame) -> pd.DataFrame:
    result = customers.copy()

    score_columns = {
        "Protect": "score_protect",
        "Re-engage": "score_reengage",
        "Develop": "score_develop",
        "Cross-sell": "score_cross_sell",
        "Promote": "score_promote",
        "Do Nothing": "score_do_nothing",
    }

    action_key = {
        "Protect": "protect",
        "Re-engage": "reengage",
        "Develop": "develop",
        "Cross-sell": "crosssell",
        "Promote": "promote",
        "Do Nothing": "do_nothing",
    }

    for action, score_col in score_columns.items():
        key = action_key[action]
        strength = result[score_col] / 100

        response_probability = ACTION_RESPONSE_BASE[action] * (0.60 + 0.80 * strength)

        if action == "Promote":
            rule_response_probability = response_probability * (
                0.70 + result["promo_responsiveness_score"]
            )
            governed_mask = (
                result["promotion_model_accepted"].fillna(False)
                & result["promotion_response_propensity"].notna()
            )
            response_probability = rule_response_probability.where(
                ~governed_mask,
                result["promotion_response_propensity"],
            )

        if action == "Cross-sell":
            rule_response_probability = response_probability * (
                0.70 + result["category_headroom_score"]
            )
            governed_mask = (
                result["cross_sell_model_accepted"].fillna(False)
                & result["cross_sell_propensity"].notna()
            )
            response_probability = rule_response_probability.where(
                ~governed_mask,
                result["cross_sell_propensity"],
            )

        response_probability = response_probability.clip(0, 0.80)

        if action == "Do Nothing":
            expected_sales = pd.Series(0.0, index=result.index)
            expected_margin = pd.Series(0.0, index=result.index)
        else:
            expected_sales = (
                result["expected_baseline_sales"]
                * ACTION_SALES_UPLIFT[action]
                * response_probability
            )
            expected_margin = (
                expected_sales * result["observed_margin_rate"]
                - ACTION_COST[action]
            )

        result[f"{key}_response_probability"] = response_probability.round(3)
        result[f"{key}_expected_incremental_sales"] = expected_sales.round(2)
        result[f"{key}_expected_incremental_margin"] = expected_margin.round(2)

    return result


# ---------------------------------------------------------------------
# Recommendation selection
# ---------------------------------------------------------------------

def select_next_best_action(customers: pd.DataFrame) -> pd.DataFrame:
    result = customers.copy()

    score_columns = {
        "Protect": "score_protect",
        "Re-engage": "score_reengage",
        "Develop": "score_develop",
        "Cross-sell": "score_cross_sell",
        "Promote": "score_promote",
        "Do Nothing": "score_do_nothing",
    }

    margin_columns = {
        "Protect": "protect_expected_incremental_margin",
        "Re-engage": "reengage_expected_incremental_margin",
        "Develop": "develop_expected_incremental_margin",
        "Cross-sell": "crosssell_expected_incremental_margin",
        "Promote": "promote_expected_incremental_margin",
        "Do Nothing": "do_nothing_expected_incremental_margin",
    }

    eligibility_columns = {
        "Protect": "eligible_protect",
        "Re-engage": "eligible_reengage",
        "Develop": "eligible_develop",
        "Cross-sell": "eligible_cross_sell",
        "Promote": "eligible_promote",
        "Do Nothing": "eligible_do_nothing",
    }

    minimum_score = {
        "Protect": 55.0,
        "Re-engage": 50.0,
        "Develop": 50.0,
        "Cross-sell": 55.0,
        "Promote": 55.0,
        "Do Nothing": 0.0,
    }

    score_frame = pd.DataFrame(
        {
            action: result[column]
            for action, column
            in score_columns.items()
        },
        index=result.index,
    )

    margin_frame = pd.DataFrame(
        {
            action: result[column]
            for action, column
            in margin_columns.items()
        },
        index=result.index,
    )

    eligible_frame = pd.DataFrame(
        {
            action: (
                result[column]
                .fillna(False)
                .astype(bool)
            )
            for action, column
            in eligibility_columns.items()
        },
        index=result.index,
    )

    valid_frame = eligible_frame.copy()

    for action in [
        "Protect",
        "Re-engage",
        "Develop",
        "Cross-sell",
        "Promote",
    ]:
        valid_frame[action] = (
            valid_frame[action]
            & margin_frame[action].gt(0)
            & score_frame[action].ge(
                minimum_score[action]
            )
        )

    valid_frame["Do Nothing"] = True

    # --------------------------------------------------------------
    # Commercial hierarchy
    #
    # 1. Prevent value destruction: Protect
    # 2. Recover value: Re-engage
    # 3. Grow healthy customers: best of Develop / Cross-sell / Promote
    # 4. Do Nothing when no relevant intervention clears the gates
    # --------------------------------------------------------------

    result["recommended_action_nba"] = "Do Nothing"

    protect_mask = valid_frame["Protect"]
    result.loc[
        protect_mask,
        "recommended_action_nba",
    ] = "Protect"

    reengage_mask = (
        ~protect_mask
        & valid_frame["Re-engage"]
    )
    result.loc[
        reengage_mask,
        "recommended_action_nba",
    ] = "Re-engage"

    growth_pool = [
        "Develop",
        "Cross-sell",
        "Promote",
    ]

    remaining_mask = (
        ~protect_mask
        & ~reengage_mask
    )

    growth_scores = (
        score_frame[growth_pool]
        .where(
            valid_frame[growth_pool],
            -1.0,
        )
    )

    best_growth_action = (
        growth_scores.idxmax(axis=1)
    )
    best_growth_score = (
        growth_scores.max(axis=1)
    )

    growth_mask = (
        remaining_mask
        & best_growth_score.ge(0)
    )

    result.loc[
        growth_mask,
        "recommended_action_nba",
    ] = best_growth_action.loc[
        growth_mask
    ]

    result["recommended_action_score"] = [
        score_frame.loc[index, action]
        for index, action
        in zip(
            result.index,
            result["recommended_action_nba"],
        )
    ]

    result["expected_incremental_margin"] = [
        margin_frame.loc[index, action]
        for index, action
        in zip(
            result.index,
            result["recommended_action_nba"],
        )
    ]

    result["expected_incremental_sales"] = 0.0
    result["response_probability"] = 0.0

    key_map = {
        "Protect": "protect",
        "Re-engage": "reengage",
        "Develop": "develop",
        "Cross-sell": "crosssell",
        "Promote": "promote",
        "Do Nothing": "do_nothing",
    }

    for action, key in key_map.items():
        mask = (
            result["recommended_action_nba"]
            .eq(action)
        )

        result.loc[
            mask,
            "expected_incremental_sales",
        ] = result.loc[
            mask,
            f"{key}_expected_incremental_sales",
        ]

        result.loc[
            mask,
            "response_probability",
        ] = result.loc[
            mask,
            f"{key}_response_probability",
        ]

    # --------------------------------------------------------------
    # Alternative = highest-scoring other VALID action.
    # If no other intervention is valid, Do Nothing is the alternative.
    # --------------------------------------------------------------

    valid_scores = (
        score_frame
        .where(
            valid_frame,
            -1.0,
        )
    )

    alternatives = []

    for index in result.index:
        selected = result.at[
            index,
            "recommended_action_nba",
        ]

        row_scores = (
            valid_scores.loc[index]
            .drop(
                labels=[selected],
                errors="ignore",
            )
            .sort_values(
                ascending=False
            )
        )

        valid_alternatives = (
            row_scores[
                row_scores >= 0
            ]
        )

        if valid_alternatives.empty:
            alternative = "Do Nothing"
        else:
            alternative = (
                valid_alternatives.index[0]
            )

        alternatives.append(
            alternative
        )

    result["alternative_action"] = (
        alternatives
    )

    result["alternative_action_score"] = [
        score_frame.loc[index, action]
        for index, action
        in zip(
            result.index,
            result["alternative_action"],
        )
    ]

    score_gap = (
        result["recommended_action_score"]
        - result["alternative_action_score"]
    ).clip(lower=0)

    evidence_score = np.select(
        [
            result["cadence_confidence"]
            .eq("High"),
            result["cadence_confidence"]
            .eq("Medium"),
        ],
        [
            1.0,
            0.70,
        ],
        default=0.40,
    )

    result["recommendation_confidence"] = (
        100
        * (
            0.45
            * (
                result[
                    "recommended_action_score"
                ]
                / 100
            )
            + 0.30
            * (
                score_gap
                / 100
            )
            .clip(0, 1)
            + 0.25
            * evidence_score
        )
    )

    result["recommendation_confidence"] = (
        result[
            "recommendation_confidence"
        ]
        .clip(0, 100)
        .round(1)
    )

    return result


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------

def identify_primary_driver(row: pd.Series) -> str:
    drivers = {
        "Customer value": float(row.get("value_score", 0) or 0),
        "Lapse risk": float(row.get("lapse_risk_score", 0) or 0),
        "Momentum deterioration": float(row.get("momentum_risk_score", 0) or 0),
        "Promotion responsiveness": float(row.get("promo_responsiveness_score", 0) or 0) * 100,
        "Category opportunity": float(row.get("category_headroom_score", 0) or 0) * 100,
    }
    return max(drivers, key=drivers.get)


def build_rationale(row: pd.Series) -> str:
    action = row["recommended_action_nba"]
    segment = row.get("cluster_name", "customer")
    value_tier = row.get("customer_value_tier", "Unknown")
    lifecycle = row.get("lifecycle_status", "Unknown")
    category = row.get("dominant_category", "relevant category")
    channel = row.get("preferred_channel", "preferred channel")

    if action == "Protect":
        return (
            f"{value_tier} value {segment} customer with {str(lifecycle).lower()} "
            f"behaviour and elevated lapse / momentum risk. Prioritise personalised "
            f"retention via {channel}."
        )
    if action == "Re-engage":
        return (
            f"Customer shows material behavioural lapse with prior commercial value. "
            f"Re-engage through {channel} using relevant {category} messaging."
        )
    if action == "Develop":
        return (
            "Active customer with development headroom. Encourage the next purchase "
            "and deepen engagement before risk increases."
        )
    if action == "Cross-sell":
        recommended_category = row.get("recommended_cross_sell_category")
        propensity = row.get("cross_sell_propensity")
        model_used = (
            bool(row.get("cross_sell_model_accepted", False))
            and pd.notna(propensity)
        )
        if pd.notna(recommended_category):
            model_phrase = (
                f" Governed 180-day cross-sell propensity is {float(propensity):.1%}."
                if model_used else ""
            )
            return (
                f"Active customer with category expansion opportunity. Current behaviour "
                f"is concentrated around {category}; recommended adjacent category is "
                f"{recommended_category}.{model_phrase}"
            )
        return (
            f"Active customer with category expansion opportunity. Current behaviour "
            f"is concentrated around {category}; test a relevant adjacent category."
        )
    if action == "Promote":
        propensity = row.get("promotion_response_propensity")
        model_used = (
            bool(row.get("promotion_model_accepted", False))
            and pd.notna(propensity)
        )
        campaign_channel = row.get("recommended_campaign_channel")
        discount_depth = row.get("recommended_discount_depth")
        offer_category = row.get("recommended_offer_category")

        chosen_channel = (
            campaign_channel
            if pd.notna(campaign_channel)
            else channel
        )
        model_phrase = (
            f" Governed promotion-response propensity is {float(propensity):.1%}."
            if model_used else ""
        )
        offer_phrase = ""
        if pd.notna(offer_category):
            offer_phrase += f" Target {offer_category}."
        if pd.notna(discount_depth):
            depth = float(discount_depth)
            depth_pct = depth * 100 if depth <= 1 else depth
            offer_phrase += f" Recommended discount depth is {depth_pct:.0f}%."

        return (
            f"Customer has positive promotion-response evidence and intervention economics."
            f"{model_phrase} Use targeted, margin-controlled activity via "
            f"{chosen_channel}.{offer_phrase}"
        )
    return (
        "No intervention currently clears the commercial threshold. Retain the customer "
        "in standard lifecycle activity and monitor for a material change in risk or value."
    )


def add_explainability(customers: pd.DataFrame) -> pd.DataFrame:
    result = customers.copy()
    result["primary_driver"] = result.apply(identify_primary_driver, axis=1)
    result["recommendation_rationale"] = result.apply(build_rationale, axis=1)
    result["economically_positive"] = result["expected_incremental_margin"].gt(0)
    result.loc[
        result["recommended_action_nba"].eq("Do Nothing"),
        "economically_positive",
    ] = False
    return result


# ---------------------------------------------------------------------
# Build full NBA table
# ---------------------------------------------------------------------

def build_next_best_action(
    priority: pd.DataFrame,
    customer_ltv: pd.DataFrame,
    clusters: pd.DataFrame,
    transactions: pd.DataFrame,
    cross_sell_scores: pd.DataFrame | None = None,
    cross_sell_recommendations: pd.DataFrame | None = None,
    cross_sell_model_accepted: bool = False,
    promotion_scores: pd.DataFrame | None = None,
    promotion_model_accepted: bool = False,
) -> pd.DataFrame:
    transaction_behaviour = build_transaction_behaviour(transactions)
    nba = build_nba_features(
        priority,
        customer_ltv,
        clusters,
        transaction_behaviour,
        cross_sell_scores=cross_sell_scores,
        cross_sell_recommendations=cross_sell_recommendations,
        cross_sell_model_accepted=cross_sell_model_accepted,
        promotion_scores=promotion_scores,
        promotion_model_accepted=promotion_model_accepted,
    )
    nba = add_action_scores(nba)
    nba = calculate_action_economics(nba)
    nba = select_next_best_action(nba)
    nba = add_explainability(nba)
    return nba


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(nba: pd.DataFrame) -> None:
    print("\nNEXT BEST ACTION QA")
    print("=" * 80)
    print(f"Resolved golden customers: {len(nba):,}")

    print(
        f"Unique golden customer IDs: "
        f"{nba['golden_customer_id'].nunique():,}"
    )
    print(
        f"Duplicate golden customer IDs: "
        f"{nba['golden_customer_id'].duplicated().sum():,}"
    )

    print("\nRecommended actions:")
    print(nba["recommended_action_nba"].value_counts(dropna=False))

    print("\nExpected incremental margin by action:")
    economics = (
        nba.groupby("recommended_action_nba")
        .agg(
            customers=("golden_customer_id", "nunique"),
            expected_incremental_sales=("expected_incremental_sales", "sum"),
            expected_incremental_margin=("expected_incremental_margin", "sum"),
            median_confidence=("recommendation_confidence", "median"),
        )
        .sort_values("expected_incremental_margin", ascending=False)
    )
    print(economics.round(1))

    do_nothing_share = nba["recommended_action_nba"].eq("Do Nothing").mean()
    print(f"\nDo Nothing share: {do_nothing_share:.1%}")

    intervention_mask = ~nba["recommended_action_nba"].eq("Do Nothing")
    positive = nba.loc[intervention_mask, "expected_incremental_margin"].gt(0).all()
    print(f"All recommended interventions economically positive: {positive}")

    print("\nTop 10 recommendations:")
    display_columns = [
        "golden_customer_id",
        "cluster_name",
        "customer_value_tier",
        "lifecycle_status",
        "recommended_action_nba",
        "recommendation_confidence",
        "expected_incremental_margin",
        "primary_driver",
        "alternative_action",
    ]
    print(
        nba[display_columns]
        .sort_values(
            ["expected_incremental_margin", "recommendation_confidence"],
            ascending=[False, False],
        )
        .head(10)
        .to_string(index=False)
    )


# ---------------------------------------------------------------------
# Governed propensity inputs
# ---------------------------------------------------------------------

def load_cross_sell_model_status() -> bool:
    if not CROSS_SELL_GOVERNANCE_FILE.exists():
        return False

    governance = pd.read_parquet(CROSS_SELL_GOVERNANCE_FILE)
    if governance.empty:
        return False

    for column in ["final_model_status", "model_status", "status"]:
        if column in governance.columns:
            values = governance[column].astype(str).str.upper()
            return values.eq("ACCEPTED").any()

    # Support the governance table produced by the modelling workflow where
    # every gate is recorded explicitly.
    if "pass" in governance.columns:
        passed = governance["pass"]
        if passed.dtype != bool:
            passed = passed.astype(str).str.lower().isin(["true", "1", "yes"])
        return bool(passed.all())

    return False


def load_promotion_model_status() -> bool:
    if not PROMOTION_GOVERNANCE_FILE.exists():
        return False

    governance = pd.read_parquet(PROMOTION_GOVERNANCE_FILE)
    if governance.empty:
        return False

    for column in ["final_model_status", "model_status", "status"]:
        if column in governance.columns:
            values = governance[column].astype(str).str.upper()
            return values.eq("ACCEPTED").any()

    if "pass" in governance.columns:
        passed = governance["pass"]
        if passed.dtype != bool:
            passed = passed.astype(str).str.lower().isin(["true", "1", "yes"])
        return bool(passed.all())

    return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Next Best Action source data...")
    priority = pd.read_parquet(PRIORITY_FILE)
    customer_ltv = pd.read_parquet(LTV_FILE)
    clusters = pd.read_parquet(CLUSTER_FILE)
    transactions = pd.read_parquet(TRANSACTION_FILE)

    cross_sell_model_accepted = load_cross_sell_model_status()
    cross_sell_scores = (
        pd.read_parquet(CROSS_SELL_SCORE_FILE)
        if CROSS_SELL_SCORE_FILE.exists()
        else None
    )
    cross_sell_recommendations = (
        pd.read_parquet(CROSS_SELL_RECOMMENDATION_FILE)
        if CROSS_SELL_RECOMMENDATION_FILE.exists()
        else None
    )

    promotion_model_accepted = load_promotion_model_status()
    promotion_scores = (
        pd.read_parquet(PROMOTION_SCORE_FILE)
        if PROMOTION_SCORE_FILE.exists()
        else None
    )

    print(
        "Cross-sell governed model status: "
        f"{'ACCEPTED' if cross_sell_model_accepted else 'FALLBACK / NOT ACCEPTED'}"
    )
    print(
        "Promotion-response governed model status: "
        f"{'ACCEPTED' if promotion_model_accepted else 'FALLBACK / NOT ACCEPTED'}"
    )

    required_inputs = {
        "Customer priority": (
            priority,
            {
                "golden_customer_id",
                "customer_value_tier",
                "rfm_segment",
                "active_customer",
                "lifecycle_status",
                "cadence_trend_status",
                "cadence_confidence",
                "trailing_12m_sales",
                "trailing_12m_margin",
                "value_score",
                "lapse_risk_score",
                "momentum_risk_score",
            },
        ),
        "Customer LTV": (
            customer_ltv,
            {"golden_customer_id"},
        ),
        "Customer clusters": (
            clusters,
            {"golden_customer_id"},
        ),
        "Golden customer transactions": (
            transactions,
            {
                "golden_customer_id",
                "order_id",
                "net_sales",
                "gross_margin",
                "discount_pct",
                "category",
                "channel",
            },
        ),
    }

    for label, (frame, required) in required_inputs.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{label} is missing required columns: "
                f"{sorted(missing)}"
            )

    for label, frame in [
        ("Customer priority", priority),
        ("Customer LTV", customer_ltv),
        ("Customer clusters", clusters),
    ]:
        if frame["golden_customer_id"].duplicated().any():
            raise ValueError(
                f"{label} must contain one row per "
                "golden_customer_id."
            )

    print("Building golden-customer Next Best Action recommendations...")
    nba = build_next_best_action(
        priority,
        customer_ltv,
        clusters,
        transactions,
        cross_sell_scores=cross_sell_scores,
        cross_sell_recommendations=cross_sell_recommendations,
        cross_sell_model_accepted=cross_sell_model_accepted,
        promotion_scores=promotion_scores,
        promotion_model_accepted=promotion_model_accepted,
    )

    nba.to_parquet(OUTPUT_FILE, index=False)
    run_qa(nba)

    print("\nFile created:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
