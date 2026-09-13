from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.filters import (
    render_customer_filters,
)


# =========================================================
# PATHS
# =========================================================

APP_ROOT = Path(__file__).resolve().parents[1]

PRIORITY_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "customer_priority.parquet"
)

CLUSTER_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "customer_clusters.parquet"
)


# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    priority = pd.read_parquet(
        PRIORITY_FILE
    )

    clusters = pd.read_parquet(
        CLUSTER_FILE
    )

    cluster_lookup = clusters[
        [
            "golden_customer_id",
            "cluster",
            "cluster_name",
        ]
    ]

    data = priority.merge(
        cluster_lookup,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    data["cluster_name"] = (
        data["cluster_name"]
        .fillna("Not Yet Clustered")
    )

    return data


df = load_data()

filtered, filters = (
    render_customer_filters(df)
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title(
    "AU Retail Customer Analytics Decision Lab"
)

st.caption(
    "Synthetic Australian retail customer scenario | "
    "Resolved Golden Customers | Behaviour → Cadence → Value → Decision"
)

st.header(
    "Decision Queue"
)

st.caption(
    "Which customers should be prioritised for retention, "
    "re-engagement or development activity?"
)


# =========================================================
# EMPTY FILTER CHECK
# =========================================================

if filtered.empty:

    st.warning(
        "No customers match the current filters."
    )

    st.stop()


# =========================================================
# DECISION FILTERS
# =========================================================

st.subheader(
    "Decision controls"
)

control1, control2, control3 = (
    st.columns(3)
)

decision_order = [
    "Protect Now",
    "Proactive Retention",
    "Re-engage",
    "Watch Closely",
    "Develop",
    "Maintain",
    "Acquisition Opportunity",
    "Low Priority",
]

available_decisions = [
    decision
    for decision in decision_order
    if decision
    in filtered[
        "decision_group"
    ].dropna().unique()
]

with control1:

    decision_options = [
        "All Priority Decisions",
        *available_decisions,
    ]

    selected_decision = st.selectbox(
        "Decision Group",
        options=decision_options,
    )


with control2:

    priority_threshold = (
        st.slider(
            "Minimum Priority Score",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
        )
    )


with control3:

    max_rows = (
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


priority_decisions = [
    "Protect Now",
    "Proactive Retention",
    "Re-engage",
    "Watch Closely",
]

if selected_decision == "All Priority Decisions":

    selected_decisions = [
        decision
        for decision in priority_decisions
        if decision in available_decisions
    ]

else:

    selected_decisions = [
        selected_decision
    ]


queue = filtered.loc[
    filtered[
        "decision_group"
    ].isin(
        selected_decisions
    )
    & (
        filtered[
            "priority_score"
        ]
        >= priority_threshold
    )
].copy()


# =========================================================
# KPI CALCULATIONS
# =========================================================

queue_customers = (
    queue[
        "golden_customer_id"
    ]
    .nunique()
)

queue_sales = (
    queue[
        "trailing_12m_sales"
    ]
    .fillna(0)
    .sum()
)

queue_margin = (
    queue[
        "trailing_12m_margin"
    ]
    .fillna(0)
    .sum()
)

median_priority = (
    queue[
        "priority_score"
    ]
    .median()
    if not queue.empty
    else 0
)

high_value_queue = int(
    queue[
        "customer_value_tier"
    ]
    .isin(
        [
            "High",
            "Very High",
        ]
    )
    .sum()
)


# =========================================================
# KPI ROW
# =========================================================

st.subheader(
    "Queue position"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

kpi1.metric(
    "Customers in Queue",
    f"{queue_customers:,}",
)

kpi2.metric(
    "High / Very High Value",
    f"{high_value_queue:,}",
)

kpi3.metric(
    "Median Priority Score",
    f"{median_priority:.1f}",
)

kpi4.metric(
    "Trailing 12M Sales",
    f"${queue_sales / 1_000_000:.2f}m",
)

kpi5.metric(
    "Trailing 12M Margin",
    f"${queue_margin / 1_000_000:.2f}m",
)


# =========================================================
# DECISION MIX + VALUE EXPOSURE
# =========================================================

st.subheader(
    "Queue composition"
)

left, right = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Decision mix
# ---------------------------------------------------------

with left:

    st.markdown(
        "**Customers by decision group**"
    )

    decision_summary = (
        queue[
            "decision_group"
        ]
        .value_counts()
        .reindex(
            decision_order,
            fill_value=0,
        )
        .rename_axis(
            "Decision Group"
        )
        .reset_index(
            name="Customers"
        )
    )

    decision_summary = (
        decision_summary.loc[
            decision_summary[
                "Customers"
            ] > 0
        ]
    )

    decision_chart = px.bar(
        decision_summary,
        x="Customers",
        y="Decision Group",
        orientation="h",
    )

    decision_chart.update_layout(
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

    decision_chart.update_yaxes(
        categoryorder="array",
        categoryarray=decision_order[::-1],
    )

    st.plotly_chart(
        decision_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Margin exposure
# ---------------------------------------------------------

with right:

    st.markdown(
        "**Trailing 12M margin by decision group**"
    )

    margin_summary = (
        queue
        .groupby(
            "decision_group",
            as_index=False,
        )
        .agg(
            trailing_12m_margin=(
                "trailing_12m_margin",
                "sum",
            ),
        )
    )

    margin_summary[
        "decision_group"
    ] = pd.Categorical(
        margin_summary[
            "decision_group"
        ],
        categories=decision_order,
        ordered=True,
    )

    margin_summary = (
        margin_summary
        .sort_values(
            "decision_group"
        )
    )

    margin_chart = px.bar(
        margin_summary,
        x="trailing_12m_margin",
        y="decision_group",
        orientation="h",
        labels={
            "decision_group":
                "Decision Group",
            "trailing_12m_margin":
                "Trailing 12M Margin",
        },
    )

    margin_chart.update_layout(
        showlegend=False,
        xaxis_title="Trailing 12M Margin",
        yaxis_title=None,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    margin_chart.update_yaxes(
        categoryorder="array",
        categoryarray=decision_order[::-1],
    )

    st.plotly_chart(
        margin_chart,
        use_container_width=True,
    )


# =========================================================
# PRIORITY MAP
# =========================================================

st.subheader(
    "Priority map"
)

st.caption(
    "Customers positioned by behavioural risk and commercial value."
)

if queue.empty:

    st.info(
        "No customers match the current decision controls."
    )

else:

    priority_map = px.scatter(
        queue,
        x="lapse_risk_score",
        y="value_score",
        size="trailing_12m_margin",
        color="decision_group",
        hover_name="golden_customer_id",
        hover_data={
            "cluster_name": True,
            "customer_value_tier": True,
            "lifecycle_status": True,
            "cadence_trend_status": True,
            "priority_score": ":.1f",
            "trailing_12m_sales": ":,.0f",
            "trailing_12m_margin": ":,.0f",
            "decision_group": True,
        },
        labels={
            "lapse_risk_score":
                "Lapse Risk Score",
            "value_score":
                "Commercial Value Score",
            "decision_group":
                "Decision Group",
            "cluster_name":
                "Customer Segment",
            "customer_value_tier":
                "Value Tier",
            "lifecycle_status":
                "Lifecycle",
            "cadence_trend_status":
                "Cadence Trend",
            "priority_score":
                "Priority Score",
            "trailing_12m_sales":
                "Trailing 12M Sales",
            "trailing_12m_margin":
                "Trailing 12M Margin",
        },
    )

    priority_map.update_layout(
        xaxis_title="Lapse Risk Score",
        yaxis_title="Commercial Value Score",
        legend_title="Decision Group",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        priority_map,
        use_container_width=True,
    )

    st.caption(
        "Customers further right have greater behavioural lapse risk. "
        "Customers higher on the chart carry greater commercial value."
    )


# =========================================================
# ACTION SUMMARY
# =========================================================

st.subheader(
    "Recommended action summary"
)

action_summary = (
    queue
    .groupby(
        [
            "decision_group",
            "recommended_action",
        ],
        as_index=False,
    )
    .agg(
        customers=(
            "golden_customer_id",
            "nunique",
        ),
        trailing_12m_sales=(
            "trailing_12m_sales",
            "sum",
        ),
        trailing_12m_margin=(
            "trailing_12m_margin",
            "sum",
        ),
        median_priority_score=(
            "priority_score",
            "median",
        ),
    )
)

if action_summary.empty:

    st.info(
        "No recommended actions match the current controls."
    )

else:

    action_summary = (
        action_summary
        .sort_values(
            [
                "decision_group",
                "median_priority_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    display_action_summary = (
        action_summary.copy()
    )

    display_action_summary[
        "trailing_12m_sales"
    ] = (
        display_action_summary[
            "trailing_12m_sales"
        ]
        .fillna(0)
        .map(
            lambda x:
                f"${x:,.0f}"
        )
    )

    display_action_summary[
        "trailing_12m_margin"
    ] = (
        display_action_summary[
            "trailing_12m_margin"
        ]
        .fillna(0)
        .map(
            lambda x:
                f"${x:,.0f}"
        )
    )

    display_action_summary[
        "median_priority_score"
    ] = (
        display_action_summary[
            "median_priority_score"
        ]
        .map(
            lambda x:
                f"{x:.1f}"
        )
    )

    display_action_summary = (
        display_action_summary.rename(
            columns={
                "decision_group":
                    "Decision Group",
                "recommended_action":
                    "Recommended Action",
                "customers":
                    "Customers",
                "trailing_12m_sales":
                    "Trailing 12M Sales",
                "trailing_12m_margin":
                    "Trailing 12M Margin",
                "median_priority_score":
                    "Median Priority Score",
            }
        )
    )

    st.dataframe(
        display_action_summary,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# PRIORITISED CUSTOMER QUEUE
# =========================================================

st.subheader(
    "Prioritised customer queue"
)

st.caption(
    "Customers ranked by priority score for CRM or customer "
    "engagement action."
)

priority_queue = (
    queue
    .sort_values(
        [
            "priority_score",
            "trailing_12m_margin",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(
        max_rows
    )
    [
        [
            "golden_customer_id",
            "cluster_name",
            "customer_value_tier",
            "lifecycle_status",
            "cadence_trend_status",
            "days_since_last_purchase",
            "median_purchase_gap_days",
            "adjusted_lapse_ratio",
            "value_score",
            "lapse_risk_score",
            "momentum_risk_score",
            "priority_score",
            "trailing_12m_sales",
            "trailing_12m_margin",
            "decision_group",
            "recommended_action",
        ]
    ]
    .copy()
)


if priority_queue.empty:

    st.info(
        "No customers match the current decision controls."
    )

else:

    export_queue = (
        priority_queue.copy()
    )

    priority_queue[
        "days_since_last_purchase"
    ] = (
        priority_queue[
            "days_since_last_purchase"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    priority_queue[
        "median_purchase_gap_days"
    ] = (
        priority_queue[
            "median_purchase_gap_days"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    priority_queue[
        "adjusted_lapse_ratio"
    ] = (
        priority_queue[
            "adjusted_lapse_ratio"
        ]
        .map(
            lambda x:
                f"{x:.2f}x"
                if pd.notna(x)
                else ""
        )
    )

    for column in [
        "value_score",
        "lapse_risk_score",
        "momentum_risk_score",
        "priority_score",
    ]:

        priority_queue[
            column
        ] = (
            priority_queue[
                column
            ]
            .map(
                lambda x:
                    f"{x:.1f}"
                    if pd.notna(x)
                    else ""
            )
        )

    priority_queue[
        "trailing_12m_sales"
    ] = (
        priority_queue[
            "trailing_12m_sales"
        ]
        .fillna(0)
        .map(
            lambda x:
                f"${x:,.0f}"
        )
    )

    priority_queue[
        "trailing_12m_margin"
    ] = (
        priority_queue[
            "trailing_12m_margin"
        ]
        .fillna(0)
        .map(
            lambda x:
                f"${x:,.0f}"
        )
    )

    priority_queue = (
        priority_queue.rename(
            columns={
                "golden_customer_id":
                    "Customer",
                "cluster_name":
                    "Customer Segment",
                "customer_value_tier":
                    "Value Tier",
                "lifecycle_status":
                    "Lifecycle",
                "cadence_trend_status":
                    "Cadence Trend",
                "days_since_last_purchase":
                    "Days Since Purchase",
                "median_purchase_gap_days":
                    "Typical Gap (Days)",
                "adjusted_lapse_ratio":
                    "Lapse Ratio",
                "value_score":
                    "Value Score",
                "lapse_risk_score":
                    "Lapse Risk Score",
                "momentum_risk_score":
                    "Momentum Risk Score",
                "priority_score":
                    "Priority Score",
                "trailing_12m_sales":
                    "Trailing 12M Sales",
                "trailing_12m_margin":
                    "Trailing 12M Margin",
                "decision_group":
                    "Decision Group",
                "recommended_action":
                    "Recommended Action",
            }
        )
    )

    st.dataframe(
        priority_queue,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# CRM EXPORT
# =========================================================

st.subheader(
    "CRM activation export"
)

if queue.empty:

    st.info(
        "No customers available to export."
    )

else:

    crm_export = (
        queue
        .sort_values(
            [
                "priority_score",
                "trailing_12m_margin",
            ],
            ascending=[
                False,
                False,
            ],
        )
        [
            [
                "golden_customer_id",
                "cluster_name",
                "customer_value_tier",
                "lifecycle_status",
                "cadence_trend_status",
                "days_since_last_purchase",
                "median_purchase_gap_days",
                "adjusted_lapse_ratio",
                "value_score",
                "lapse_risk_score",
                "momentum_risk_score",
                "priority_score",
                "trailing_12m_sales",
                "trailing_12m_margin",
                "decision_group",
                "recommended_action",
            ]
        ]
        .rename(
            columns={
                "cluster_name":
                    "customer_segment",
                "customer_value_tier":
                    "value_tier",
                "lifecycle_status":
                    "lifecycle_status",
                "cadence_trend_status":
                    "cadence_trend",
                "days_since_last_purchase":
                    "days_since_purchase",
                "median_purchase_gap_days":
                    "typical_gap_days",
                "adjusted_lapse_ratio":
                    "lapse_ratio",
                "value_score":
                    "value_score",
                "lapse_risk_score":
                    "lapse_risk_score",
                "momentum_risk_score":
                    "momentum_risk_score",
                "priority_score":
                    "priority_score",
                "trailing_12m_sales":
                    "trailing_12m_sales",
                "trailing_12m_margin":
                    "trailing_12m_margin",
                "decision_group":
                    "decision_group",
                "recommended_action":
                    "recommended_action",
            }
        )
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label="Download CRM Decision Queue",
        data=crm_export,
        file_name="customer_decision_queue.csv",
        mime="text/csv",
    )

    st.caption(
        "The exported file contains all customers matching the "
        "current sidebar filters and decision controls, not only "
        "the rows currently visible in the on-screen queue."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Priority score combines commercial value, behavioural lapse "
    "risk and momentum risk. Decision groups translate those signals "
    "into operational customer actions. The queue is intended to "
    "support CRM prioritisation and analyst review rather than "
    "automated customer contact without business oversight."
)