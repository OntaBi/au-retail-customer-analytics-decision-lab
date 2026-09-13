from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.filters import render_customer_filters


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]

NBA_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "customer_next_best_action.parquet"
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ACTION_ORDER = [
    "Protect",
    "Re-engage",
    "Develop",
    "Cross-sell",
    "Promote",
    "Do Nothing",
]

ACTION_COLORS = {
    "Protect": "#F28E8B",
    "Re-engage": "#F4A261",
    "Develop": "#E9C46A",
    "Cross-sell": "#73C0A8",
    "Promote": "#5DADE2",
    "Do Nothing": "#8B8F99",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:

    nba = pd.read_parquet(
        NBA_FILE
    )

    required_defaults = {
        "cluster_name":
            "Not Yet Clustered",
        "customer_value_tier":
            "Unknown",
        "recommended_action_nba":
            "Do Nothing",
        "recommendation_confidence":
            0.0,
        "expected_incremental_margin":
            0.0,
        "expected_incremental_sales":
            0.0,
        "response_probability":
            0.0,
        "primary_driver":
            "Insufficient evidence",
        "alternative_action":
            "Do Nothing",
        "recommendation_rationale":
            "No recommendation rationale available.",
        "dominant_category":
            "No dominant category",
        "preferred_channel":
            "No clear channel",
        "cross_sell_model_accepted":
            False,
        "cross_sell_propensity":
            np.nan,
        "cross_sell_propensity_source":
            "Decision-engine fallback",
        "recommended_cross_sell_category":
            "No category recommendation",
        "cross_sell_category_confidence":
            "Not available",
        "promotion_model_accepted":
            False,
        "promotion_response_propensity":
            np.nan,
        "promotion_propensity_source":
            "Decision-engine fallback",
        "recommended_campaign_channel":
            "No campaign channel",
        "recommended_discount_depth":
            np.nan,
        "recommended_offer_category":
            "No offer category",
        "promotion_response_band":
            "Not available",
    }

    for column, default in required_defaults.items():

        if column not in nba.columns:
            nba[column] = default

        nba[column] = (
            nba[column]
            .fillna(default)
        )

    return nba


nba = load_data()

filtered_customers, filters = (
    render_customer_filters(
        nba
    )
)


# ---------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------

st.title(
    "AU Retail Customer Analytics Decision Lab"
)

st.caption(
    "Synthetic Australian retail customer scenario | "
    "Resolved Golden Customers | Behaviour → Cadence → Value → Decision"
)

st.header(
    "Next Best Action"
)

st.caption(
    "Which customers should we act on, what action should we take, "
    "and does intervention create enough expected value to justify it?"
)


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

if filtered_customers.empty:

    st.warning(
        "No customers match the current filters."
    )

    st.stop()


# ---------------------------------------------------------------------
# Page-specific controls
# ---------------------------------------------------------------------

st.subheader(
    "Recommendation controls"
)

control1, control2, control3 = (
    st.columns(
        [1.5, 1.2, 1.0]
    )
)

with control1:

    selected_action = (
        st.selectbox(
            "Recommended Action",
            options=[
                "All",
                *ACTION_ORDER,
            ],
            index=0,
        )
    )

with control2:

    minimum_confidence = (
        st.slider(
            "Minimum Confidence",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )
    )

with control3:

    queue_size = (
        st.selectbox(
            "Queue Size",
            options=[
                25,
                50,
                100,
                250,
                500,
            ],
            index=2,
        )
    )


page_df = filtered_customers.copy()

if selected_action != "All":

    page_df = page_df.loc[
        page_df[
            "recommended_action_nba"
        ].eq(selected_action)
    ]

page_df = page_df.loc[
    page_df[
        "recommendation_confidence"
    ]
    .fillna(0)
    .ge(
        minimum_confidence
    )
].copy()


if page_df.empty:

    st.info(
        "No customers match the current recommendation controls."
    )

    st.stop()


# ---------------------------------------------------------------------
# Portfolio KPIs
# ---------------------------------------------------------------------

intervention_mask = (
    ~page_df[
        "recommended_action_nba"
    ]
    .eq("Do Nothing")
)

recommended_interventions = int(
    intervention_mask.sum()
)

do_nothing_customers = int(
    (
        page_df[
            "recommended_action_nba"
        ]
        .eq("Do Nothing")
    )
    .sum()
)

do_nothing_share = (
    do_nothing_customers
    / len(page_df)
    if len(page_df) > 0
    else 0
)

expected_incremental_margin = (
    page_df.loc[
        intervention_mask,
        "expected_incremental_margin",
    ]
    .sum()
)

expected_incremental_sales = (
    page_df.loc[
        intervention_mask,
        "expected_incremental_sales",
    ]
    .sum()
)

average_confidence = (
    page_df.loc[
        intervention_mask,
        "recommendation_confidence",
    ]
    .mean()
)

if pd.isna(
    average_confidence
):
    average_confidence = 0.0


st.subheader(
    "Portfolio opportunity"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

kpi1.metric(
    "Recommended Interventions",
    f"{recommended_interventions:,}",
)

kpi2.metric(
    "Expected Incremental Margin",
    f"${expected_incremental_margin:,.0f}",
)

kpi3.metric(
    "Expected Incremental Sales",
    f"${expected_incremental_sales:,.0f}",
)

kpi4.metric(
    "Average Confidence",
    f"{average_confidence:.1f}%",
)

kpi5.metric(
    "Do Nothing",
    f"{do_nothing_share:.1%}",
)

st.caption(
    "Interventions are recommended only when the action is relevant "
    "and expected incremental margin is positive. Do Nothing is an "
    "explicit commercial decision, not a missing recommendation."
)


# ---------------------------------------------------------------------
# Decision intelligence governance
# ---------------------------------------------------------------------

st.subheader("Decision intelligence governance")

cross_sell_accepted = bool(
    nba["cross_sell_model_accepted"].fillna(False).any()
)
promotion_accepted = bool(
    nba["promotion_model_accepted"].fillna(False).any()
)

gov1, gov2, gov3 = st.columns(3)

gov1.metric(
    "Cross-sell Propensity",
    "Accepted" if cross_sell_accepted else "Fallback",
)
gov2.metric(
    "Promotion Response",
    "Accepted" if promotion_accepted else "Fallback",
)
gov3.metric(
    "Re-engagement Propensity",
    "Rejected → Rules",
)

st.info(
    "Accepted propensity models inform customer decisioning. "
    "Rejected models automatically fall back to governed decision rules."
)

st.caption(
    "Accepted models inform action relevance and/or response probability; "
    "customer eligibility and positive expected incremental economics remain "
    "mandatory before an intervention is recommended."
)


# ---------------------------------------------------------------------
# Action mix and economics
# ---------------------------------------------------------------------

st.subheader(
    "Recommendation portfolio"
)

left, right = st.columns(2)

action_summary = (
    page_df
    .groupby(
        "recommended_action_nba",
        dropna=False,
    )
    .agg(
        customers=(
            "golden_customer_id",
            "nunique",
        ),
        expected_incremental_margin=(
            "expected_incremental_margin",
            "sum",
        ),
        median_confidence=(
            "recommendation_confidence",
            "median",
        ),
    )
    .reset_index()
)

action_summary[
    "recommended_action_nba"
] = pd.Categorical(
    action_summary[
        "recommended_action_nba"
    ],
    categories=ACTION_ORDER,
    ordered=True,
)

action_summary = (
    action_summary
    .sort_values(
        "recommended_action_nba"
    )
)

with left:

    st.markdown(
        "**Customers by recommended action**"
    )

    action_mix_chart = px.bar(
        action_summary,
        x="customers",
        y="recommended_action_nba",
        orientation="h",
        color="recommended_action_nba",
        category_orders={
            "recommended_action_nba":
                ACTION_ORDER
        },
        color_discrete_map=ACTION_COLORS,
        labels={
            "customers":
                "Customers",
            "recommended_action_nba":
                "Recommended Action",
        },
    )

    action_mix_chart.update_layout(
        showlegend=False,
        xaxis_title="Customers",
        yaxis_title=None,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        action_mix_chart,
        use_container_width=True,
    )


with right:

    st.markdown(
        "**Expected incremental margin by action**"
    )

    economics_summary = (
        action_summary.loc[
            ~action_summary[
                "recommended_action_nba"
            ]
            .eq("Do Nothing")
        ]
        .copy()
    )

    economics_chart = px.bar(
        economics_summary,
        x="expected_incremental_margin",
        y="recommended_action_nba",
        orientation="h",
        color="recommended_action_nba",
        category_orders={
            "recommended_action_nba":
                ACTION_ORDER
        },
        color_discrete_map=ACTION_COLORS,
        labels={
            "expected_incremental_margin":
                "Expected Incremental Margin",
            "recommended_action_nba":
                "Recommended Action",
        },
    )

    economics_chart.update_layout(
        showlegend=False,
        xaxis_title="Expected Incremental Margin",
        yaxis_title=None,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    economics_chart.update_xaxes(
        tickprefix="$",
        separatethousands=True,
    )

    st.plotly_chart(
        economics_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Recommendation map
# ---------------------------------------------------------------------

st.subheader(
    "Recommendation map"
)

st.caption(
    "Customers positioned by behavioural risk and commercial value. "
    "Bubble size represents expected incremental margin for the "
    "recommended action."
)

map_df = page_df.copy()

# Keep portfolio calculations on the full filtered population, but
# sample the map for readability when the customer universe is large.
MAP_SAMPLE_SIZE = 5000

if len(map_df) > MAP_SAMPLE_SIZE:

    action_groups = []

    for action, action_df in map_df.groupby(
        "recommended_action_nba",
        dropna=False,
    ):

        action_share = (
            len(action_df)
            / len(map_df)
        )

        action_n = max(
            1,
            int(
                round(
                    MAP_SAMPLE_SIZE
                    * action_share
                )
            ),
        )

        action_n = min(
            action_n,
            len(action_df),
        )

        action_groups.append(
            action_df.sample(
                n=action_n,
                random_state=42,
            )
        )

    map_df = (
        pd.concat(
            action_groups,
            ignore_index=True,
        )
        .head(
            MAP_SAMPLE_SIZE
        )
    )

map_df[
    "bubble_margin"
] = (
    map_df[
        "expected_incremental_margin"
    ]
    .clip(
        lower=0
    )
)

map_df[
    "bubble_size"
] = (
    np.sqrt(
        map_df[
            "bubble_margin"
        ]
        + 1
    )
)

recommendation_map = px.scatter(
    map_df,
    x="lapse_risk_score",
    y="value_score",
    size="bubble_size",
    color="recommended_action_nba",
    color_discrete_map=ACTION_COLORS,
    category_orders={
        "recommended_action_nba":
            ACTION_ORDER
    },
    hover_data={
        "golden_customer_id":
            True,
        "cluster_name":
            True,
        "customer_value_tier":
            True,
        "recommended_action_nba":
            True,
        "recommendation_confidence":
            ":.1f",
        "expected_incremental_margin":
            ":,.2f",
        "bubble_size":
            False,
    },
    labels={
        "lapse_risk_score":
            "Lapse Risk Score",
        "value_score":
            "Commercial Value Score",
        "recommended_action_nba":
            "Recommended Action",
    },
)

recommendation_map.update_layout(
    legend=dict(
        title=None,
        orientation="h",
        yanchor="top",
        y=-0.16,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=90,
    ),
)

st.plotly_chart(
    recommendation_map,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Model evidence helpers
# ---------------------------------------------------------------------

def decision_intelligence_source(row: pd.Series) -> str:
    action = str(row.get("recommended_action_nba", "Do Nothing"))

    if (
        action == "Cross-sell"
        and bool(row.get("cross_sell_model_accepted", False))
        and pd.notna(row.get("cross_sell_propensity"))
    ):
        return "Model-informed"

    if (
        action == "Promote"
        and bool(row.get("promotion_model_accepted", False))
        and pd.notna(row.get("promotion_response_propensity"))
    ):
        return "Model-informed"

    if action == "Re-engage":
        return "Rules-informed"

    return "Decision rules"


def action_propensity(row: pd.Series):
    action = str(row.get("recommended_action_nba", "Do Nothing"))

    if action == "Cross-sell":
        value = row.get("cross_sell_propensity")
        return float(value) if pd.notna(value) else np.nan

    if action == "Promote":
        value = row.get("promotion_response_propensity")
        return float(value) if pd.notna(value) else np.nan

    return np.nan


def action_treatment(row: pd.Series) -> str:
    action = str(row.get("recommended_action_nba", "Do Nothing"))

    if action == "Cross-sell":
        category = row.get("recommended_cross_sell_category")
        if pd.notna(category) and str(category).strip():
            return f"Category: {category}"
        return "Cross-sell opportunity"

    if action == "Promote":
        pieces = []
        category = row.get("recommended_offer_category")
        channel = row.get("recommended_campaign_channel")
        depth = row.get("recommended_discount_depth")

        if pd.notna(category) and str(category).strip():
            pieces.append(str(category))
        if pd.notna(channel) and str(channel).strip():
            pieces.append(str(channel))
        if pd.notna(depth):
            depth = float(depth)
            depth_pct = depth * 100 if depth <= 1 else depth
            pieces.append(f"{depth_pct:.0f}%")

        return " | ".join(pieces) if pieces else "Promotion opportunity"

    if action == "Re-engage":
        return "Behavioural rules fallback"

    return str(row.get("primary_driver", "Decision rules"))


# ---------------------------------------------------------------------
# Prioritised intervention queue
# ---------------------------------------------------------------------

st.subheader(
    "Prioritised intervention queue"
)

st.info(
    "Activation rule: recommend intervention only where customer "
    "eligibility, action relevance and positive expected incremental "
    "economics are satisfied. Otherwise, Do Nothing remains the "
    "commercial decision."
)

st.caption(
    "Customers ranked by expected incremental margin and recommendation "
    "confidence for CRM or analyst activation."
)

queue = (
    page_df.loc[
        ~page_df[
            "recommended_action_nba"
        ]
        .eq("Do Nothing")
    ]
    .sort_values(
        [
            "expected_incremental_margin",
            "recommendation_confidence",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        queue_size
    )
    .copy()
)

if queue.empty:

    st.info(
        "No intervention recommendations match the current filters."
    )

else:

    queue["decision_intelligence"] = queue.apply(
        decision_intelligence_source,
        axis=1,
    )
    queue["action_propensity"] = queue.apply(
        action_propensity,
        axis=1,
    )
    queue["treatment_evidence"] = queue.apply(
        action_treatment,
        axis=1,
    )

    queue_display = (
        queue[
            [
                "golden_customer_id",
                "cluster_name",
                "customer_value_tier",
                "lifecycle_status",
                "recommended_action_nba",
                "decision_intelligence",
                "action_propensity",
                "recommendation_confidence",
                "expected_incremental_margin",
                "response_probability",
                "treatment_evidence",
                "primary_driver",
                "alternative_action",
            ]
        ]
        .copy()
    )

    queue_display[
        "action_propensity"
    ] = (
        queue_display[
            "action_propensity"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
                if pd.notna(x)
                else "—"
        )
    )

    queue_display[
        "recommendation_confidence"
    ] = (
        queue_display[
            "recommendation_confidence"
        ]
        .map(
            lambda x:
                f"{x:.1f}%"
        )
    )

    queue_display[
        "expected_incremental_margin"
    ] = (
        queue_display[
            "expected_incremental_margin"
        ]
        .map(
            lambda x:
                f"${x:,.0f}"
        )
    )

    queue_display[
        "response_probability"
    ] = (
        queue_display[
            "response_probability"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )

    queue_display = (
        queue_display.rename(
            columns={
                "golden_customer_id":
                    "Customer",
                "cluster_name":
                    "Customer Segment",
                "customer_value_tier":
                    "Value Tier",
                "lifecycle_status":
                    "Lifecycle",
                "recommended_action_nba":
                    "Next Best Action",
                "decision_intelligence":
                    "Decision Intelligence",
                "action_propensity":
                    "Propensity",
                "treatment_evidence":
                    "Treatment Evidence",
                "recommendation_confidence":
                    "Confidence",
                "expected_incremental_margin":
                    "Expected Incremental Margin",
                "response_probability":
                    "Response Probability",
                "primary_driver":
                    "Primary Driver",
                "alternative_action":
                    "Alternative",
            }
        )
    )

    st.dataframe(
        queue_display,
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Customer recommendation explorer
# ---------------------------------------------------------------------

st.subheader(
    "Recommendation explorer"
)

explorer_candidates = (
    queue[
        "golden_customer_id"
    ]
    .astype(str)
    .tolist()
)

if not explorer_candidates:

    explorer_candidates = (
        page_df[
            "golden_customer_id"
        ]
        .astype(str)
        .tolist()
    )

selected_customer = (
    st.selectbox(
        "Golden Customer",
        options=explorer_candidates,
        index=0,
    )
)

customer = (
    page_df.loc[
        page_df[
            "golden_customer_id"
        ]
        .astype(str)
        .eq(
            selected_customer
        )
    ]
    .iloc[0]
)


profile1, profile2, profile3, profile4 = (
    st.columns(4)
)

profile1.metric(
    "Next Best Action",
    customer[
        "recommended_action_nba"
    ],
)

profile2.metric(
    "Confidence",
    f"{customer['recommendation_confidence']:.1f}%",
)

profile3.metric(
    "Expected Incremental Margin",
    f"${customer['expected_incremental_margin']:,.0f}",
)

profile4.metric(
    "Alternative",
    customer[
        "alternative_action"
    ],
)


st.markdown(
    "**Recommendation rationale**"
)

st.info(
    customer[
        "recommendation_rationale"
    ]
)

customer_intelligence = decision_intelligence_source(customer)
customer_propensity = action_propensity(customer)
customer_treatment = action_treatment(customer)

st.markdown("**Decision intelligence**")

intel1, intel2, intel3 = st.columns(3)

intel1.metric(
    "Decision Source",
    customer_intelligence,
)

intel2.metric(
    "Relevant Propensity",
    f"{customer_propensity:.1%}"
    if pd.notna(customer_propensity)
    else "Not applicable",
)

intel3.metric(
    "Treatment / Opportunity",
    customer_treatment,
)

if customer["recommended_action_nba"] == "Cross-sell":
    st.caption(
        "Cross-sell is informed by the governance-accepted 180-day propensity "
        "model and category recommendation layer. The action still requires "
        "positive expected incremental margin."
    )
elif customer["recommended_action_nba"] == "Promote":
    st.caption(
        "Promotion is informed by the governance-accepted 30-day promotion "
        "response model. Campaign treatment evidence is used alongside "
        "commercial economics before activation."
    )
elif customer["recommended_action_nba"] == "Re-engage":
    st.caption(
        "Re-engagement remains rules-informed because the propensity model "
        "failed its governance thresholds. The rejected model is not used "
        "for operational customer decisioning."
    )
else:
    st.caption(
        "This action is driven by transparent customer-value, lifecycle, "
        "behavioural-risk and commercial decision rules."
    )


detail1, detail2, detail3 = (
    st.columns(3)
)

with detail1:

    st.markdown(
        "**Customer context**"
    )

    st.write(
        f"**Segment:** "
        f"{customer['cluster_name']}"
    )

    st.write(
        f"**Value Tier:** "
        f"{customer['customer_value_tier']}"
    )

    st.write(
        f"**Lifecycle:** "
        f"{customer['lifecycle_status']}"
    )

    st.write(
        f"**Primary Driver:** "
        f"{customer['primary_driver']}"
    )


with detail2:

    st.markdown(
        "**Behavioural evidence**"
    )

    st.write(
        f"**Lapse Risk:** "
        f"{customer['lapse_risk_score']:.1f} / 100"
    )

    st.write(
        f"**Momentum Risk:** "
        f"{customer['momentum_risk_score']:.1f} / 100"
    )

    st.write(
        f"**Value Score:** "
        f"{customer['value_score']:.1f} / 100"
    )

    st.write(
        f"**Cadence:** "
        f"{customer.get('cadence_trend_status', 'Unknown')}"
    )


with detail3:

    st.markdown(
        "**Treatment context**"
    )

    st.write(
        f"**Dominant Category:** "
        f"{customer['dominant_category']}"
    )

    st.write(
        f"**Preferred Channel:** "
        f"{customer['preferred_channel']}"
    )

    st.write(
        f"**Response Probability:** "
        f"{customer['response_probability']:.1%}"
    )

    st.write(
        f"**Expected Incremental Sales:** "
        f"${customer['expected_incremental_sales']:,.0f}"
    )

    if customer["recommended_action_nba"] == "Cross-sell":
        category = customer.get(
            "recommended_cross_sell_category",
            "No category recommendation",
        )
        confidence = customer.get(
            "cross_sell_category_confidence",
            "Not available",
        )
        st.write(f"**Recommended Category:** {category}")
        st.write(f"**Category Confidence:** {confidence}")

    if customer["recommended_action_nba"] == "Promote":
        campaign_channel = customer.get(
            "recommended_campaign_channel",
            "No campaign channel",
        )
        offer_category = customer.get(
            "recommended_offer_category",
            "No offer category",
        )
        response_band = customer.get(
            "promotion_response_band",
            "Not available",
        )
        discount_depth = customer.get(
            "recommended_discount_depth",
            np.nan,
        )

        st.write(f"**Campaign Channel:** {campaign_channel}")
        st.write(f"**Offer Category:** {offer_category}")
        st.write(f"**Response Band:** {response_band}")

        if pd.notna(discount_depth):
            discount_depth = float(discount_depth)
            discount_pct = (
                discount_depth * 100
                if discount_depth <= 1
                else discount_depth
            )
            st.write(
                f"**Recommended Discount:** {discount_pct:.0f}%"
            )


# ---------------------------------------------------------------------
# Intervene vs Do Nothing
# ---------------------------------------------------------------------

st.subheader(
    "Intervene vs Do Nothing"
)

comparison = pd.DataFrame(
    {
        "Decision": [
            customer[
                "recommended_action_nba"
            ],
            "Do Nothing",
        ],
        "Expected Incremental Margin": [
            max(
                0.0,
                float(
                    customer[
                        "expected_incremental_margin"
                    ]
                ),
            ),
            0.0,
        ],
    }
)

comparison_chart = px.bar(
    comparison,
    x="Decision",
    y="Expected Incremental Margin",
    text_auto=".0f",
    color="Decision",
    color_discrete_map={
        customer[
            "recommended_action_nba"
        ]:
            ACTION_COLORS.get(
                customer[
                    "recommended_action_nba"
                ],
                "#73C0A8",
            ),
        "Do Nothing":
            ACTION_COLORS[
                "Do Nothing"
            ],
    },
)

comparison_chart.update_layout(
    showlegend=False,
    xaxis_title=None,
    yaxis_title="Expected Incremental Margin",
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
)

comparison_chart.update_yaxes(
    tickprefix="$",
    separatethousands=True,
)

comparison_chart.update_traces(
    texttemplate="$%{y:,.0f}",
    textposition="outside",
)

st.plotly_chart(
    comparison_chart,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Action summary
# ---------------------------------------------------------------------

st.subheader(
    "Recommended action summary"
)

summary_table = (
    page_df
    .groupby(
        "recommended_action_nba"
    )
    .agg(
        customers=(
            "golden_customer_id",
            "nunique",
        ),
        expected_incremental_sales=(
            "expected_incremental_sales",
            "sum",
        ),
        expected_incremental_margin=(
            "expected_incremental_margin",
            "sum",
        ),
        median_confidence=(
            "recommendation_confidence",
            "median",
        ),
    )
    .reset_index()
)

summary_table[
    "recommended_action_nba"
] = pd.Categorical(
    summary_table[
        "recommended_action_nba"
    ],
    categories=ACTION_ORDER,
    ordered=True,
)

summary_table = (
    summary_table
    .sort_values(
        "recommended_action_nba"
    )
)

summary_display = (
    summary_table.copy()
)

summary_display[
    "expected_incremental_sales"
] = (
    summary_display[
        "expected_incremental_sales"
    ]
    .map(
        lambda x:
            f"${x:,.0f}"
    )
)

summary_display[
    "expected_incremental_margin"
] = (
    summary_display[
        "expected_incremental_margin"
    ]
    .map(
        lambda x:
            f"${x:,.0f}"
    )
)

summary_display[
    "median_confidence"
] = (
    summary_display[
        "median_confidence"
    ]
    .map(
        lambda x:
            f"{x:.1f}%"
    )
)

summary_display = (
    summary_display.rename(
        columns={
            "recommended_action_nba":
                "Next Best Action",
            "customers":
                "Customers",
            "expected_incremental_sales":
                "Expected Incremental Sales",
            "expected_incremental_margin":
                "Expected Incremental Margin",
            "median_confidence":
                "Median Confidence",
        }
    )
)

st.dataframe(
    summary_display,
    hide_index=True,
    use_container_width=True,
)


# ---------------------------------------------------------------------
# CRM export
# ---------------------------------------------------------------------

st.subheader(
    "CRM activation export"
)

export_columns = [
    "golden_customer_id",
    "cluster_name",
    "customer_value_tier",
    "lifecycle_status",
    "recommended_action_nba",
    "recommendation_confidence",
    "response_probability",
    "cross_sell_model_accepted",
    "cross_sell_propensity",
    "cross_sell_propensity_source",
    "recommended_cross_sell_category",
    "cross_sell_category_confidence",
    "promotion_model_accepted",
    "promotion_response_propensity",
    "promotion_propensity_source",
    "recommended_campaign_channel",
    "recommended_discount_depth",
    "recommended_offer_category",
    "promotion_response_band",
    "expected_incremental_sales",
    "expected_incremental_margin",
    "primary_driver",
    "alternative_action",
    "dominant_category",
    "preferred_channel",
    "recommendation_rationale",
]

available_export_columns = [
    column
    for column
    in export_columns
    if column in page_df.columns
]

crm_export = (
    page_df.loc[
        ~page_df[
            "recommended_action_nba"
        ]
        .eq("Do Nothing"),
        available_export_columns,
    ]
    .sort_values(
        [
            "expected_incremental_margin",
            "recommendation_confidence",
        ],
        ascending=[
            False,
            False,
        ],
    )
)

csv_data = (
    crm_export
    .to_csv(
        index=False
    )
    .encode(
        "utf-8"
    )
)

st.download_button(
    label="Download CRM Next Best Action Queue",
    data=csv_data,
    file_name=(
        "customer_next_best_action_queue.csv"
    ),
    mime="text/csv",
)

st.caption(
    "The export contains all intervention recommendations matching the "
    "current sidebar and recommendation controls, not only the rows "
    "visible in the on-screen queue."
)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "Next Best Action uses a governed hybrid decision architecture. "
    "Governance-accepted propensity models can inform action relevance and "
    "response probability, while rejected models automatically fall back to "
    "transparent decision rules. Customer eligibility and positive expected "
    "incremental margin remain mandatory; otherwise Do Nothing remains a "
    "valid commercial decision."
)
