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
            "customer_id",
            "cluster",
            "cluster_name",
        ]
    ]

    data = priority.merge(
        cluster_lookup,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    return data


df = load_data()

df["cluster_name"] = (
    df["cluster_name"]
    .fillna("Not Yet Clustered")
)

filtered, filters = (
    render_customer_filters(df)
)


# =========================================================
# PAGE HEADER
# =========================================================

st.title(
    "Executive Overview"
)

st.caption(
    "Customer health, commercial value and "
    "priority actions as at 31 July 2026."
)


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = (
    filtered[
        "customer_id"
    ]
    .nunique()
)

active_customers = int(
    filtered[
        "active_customer"
    ]
    .fillna(False)
    .sum()
)

active_rate = (
    active_customers
    / total_customers
    if total_customers > 0
    else 0
)

protect_now = int(
    filtered[
        "decision_group"
    ]
    .eq(
        "Protect Now"
    )
    .sum()
)

proactive_retention = int(
    filtered[
        "decision_group"
    ]
    .eq(
        "Proactive Retention"
    )
    .sum()
)

priority_customers = (
    protect_now
    + proactive_retention
)

priority_margin = (
    filtered.loc[
        filtered[
            "decision_group"
        ].isin(
            [
                "Protect Now",
                "Proactive Retention",
            ]
        ),
        "trailing_12m_margin",
    ]
    .fillna(0)
    .sum()
)

trailing_sales = (
    filtered[
        "trailing_12m_sales"
    ]
    .fillna(0)
    .sum()
)

trailing_margin = (
    filtered[
        "trailing_12m_margin"
    ]
    .fillna(0)
    .sum()
)


# =========================================================
# KPI ROW
# =========================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

with kpi1:
    st.metric(
        "Customers",
        f"{total_customers:,.0f}",
    )

with kpi2:
    st.metric(
        "Active Customers",
        f"{active_customers:,.0f}",
    )
    st.caption(
        f"{active_customers / total_customers:.1%} of total customers"
    )

with kpi3:
    st.metric(
        "Protect / Retain",
        f"{priority_customers:,.0f}",
    )

with kpi4:
    st.metric(
        "Trailing 12M Sales",
        f"${trailing_sales / 1_000_000:.1f}m",
    )

with kpi5:
    st.metric(
        "Trailing 12M Margin",
        f"${trailing_margin / 1_000_000:.1f}m",
    )


st.markdown("---")


# =========================================================
# CUSTOMER HEALTH + DECISION PRIORITY
# =========================================================

left, right = st.columns(
    [1.15, 1]
)


with left:

    st.subheader(
        "Customer Health"
    )

    lifecycle_order = [
        "On Cadence",
        "Watch",
        "At Risk",
        "Highly Lapsed",
        "Active / Limited History",
        "Inactive / Limited History",
        "Acquired / Never Purchased",
    ]

    lifecycle = (
        filtered[
            "lifecycle_status"
        ]
        .value_counts()
        .reindex(
            lifecycle_order,
            fill_value=0,
        )
        .rename_axis(
            "lifecycle_status"
        )
        .reset_index(
            name="customers"
        )
    )

    fig_lifecycle = px.bar(
        lifecycle,
        x="customers",
        y="lifecycle_status",
        orientation="h",
        labels={
            "lifecycle_status":
                "Lifecycle Status",
            "customers":
                "Customers",
        },
    )

    fig_lifecycle.update_layout(
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

    fig_lifecycle.update_yaxes(
        categoryorder="array",
        categoryarray=lifecycle_order[::-1],
    )

    st.plotly_chart(
        fig_lifecycle,
        use_container_width=True,
    )


with right:

    st.subheader(
        "Decision Priorities"
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

    decisions = (
        filtered[
            "decision_group"
        ]
        .value_counts()
        .rename_axis(
            "decision_group"
        )
        .reset_index(
            name="customers"
        )
    )

    decisions[
        "decision_group"
    ] = pd.Categorical(
        decisions[
            "decision_group"
        ],
        categories=decision_order,
        ordered=True,
    )

    decisions = (
        decisions
        .sort_values(
            "decision_group"
        )
    )

    fig_decision = px.bar(
        decisions,
        x="customers",
        y="decision_group",
        orientation="h",
        labels={
            "decision_group":
                "Decision Group",
            "customers":
                "Customers",
        },
    )

    fig_decision.update_layout(
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
        fig_decision,
        use_container_width=True,
    )


# =========================================================
# COMMERCIAL EXPOSURE
# =========================================================

st.subheader(
    "Commercial Value by Decision Group"
)

commercial = (
    filtered
    .groupby(
        "decision_group",
        as_index=False,
    )
    .agg(
        customers=(
            "customer_id",
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
    )
)

commercial[
    "decision_group"
] = pd.Categorical(
    commercial[
        "decision_group"
    ],
    categories=decision_order,
    ordered=True,
)

commercial = (
    commercial
    .sort_values(
        "decision_group"
    )
)

fig_commercial = px.bar(
    commercial,
    x="decision_group",
    y="trailing_12m_margin",
    hover_data=[
        "customers",
        "trailing_12m_sales",
    ],
    labels={
        "decision_group":
            "Decision Group",
        "trailing_12m_margin":
            "Trailing 12M Margin",
        "trailing_12m_sales":
            "Trailing 12M Sales",
        "customers":
            "Customers",
    },
)

fig_commercial.update_layout(
    showlegend=False,
    xaxis_title=None,
    yaxis_title="Trailing 12M Margin",
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
)

st.plotly_chart(
    fig_commercial,
    use_container_width=True,
)


# =========================================================
# SEGMENT VIEW
# =========================================================

st.subheader(
    "Behavioural Customer Segments"
)

segments = (
    filtered
    .groupby(
        "cluster_name",
        as_index=False,
    )
    .agg(
        customers=(
            "customer_id",
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

segments[
    "customer_share"
] = (
    segments[
        "customers"
    ]
    / segments[
        "customers"
    ].sum()
)

segments = (
    segments
    .sort_values(
        "customers",
        ascending=False,
    )
)

fig_segments = px.bar(
    segments,
    x="cluster_name",
    y="customers",
    hover_data={
        "customer_share": ":.1%",
        "trailing_12m_sales": ":,.0f",
        "trailing_12m_margin": ":,.0f",
        "median_priority_score": ":.1f",
    },
    labels={
        "cluster_name":
            "Customer Segment",
        "customers":
            "Customers",
        "customer_share":
            "Customer Share",
        "trailing_12m_sales":
            "Trailing 12M Sales",
        "trailing_12m_margin":
            "Trailing 12M Margin",
        "median_priority_score":
            "Median Priority Score",
    },
)

fig_segments.update_layout(
    showlegend=False,
    xaxis_title=None,
    yaxis_title="Customers",
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
)

st.plotly_chart(
    fig_segments,
    use_container_width=True,
)


# =========================================================
# EXECUTIVE SIGNALS
# =========================================================

st.subheader(
    "Executive Signals"
)

signal1, signal2, signal3 = (
    st.columns(3)
)


with signal1:

    highly_lapsed = (
        filtered[
            "lifecycle_status"
        ]
        .eq(
            "Highly Lapsed"
        )
        .sum()
    )

    st.metric(
        "Highly Lapsed",
        f"{highly_lapsed:,.0f}",
    )

    st.caption(
        "Customers materially beyond their "
        "expected personal purchase cadence."
    )


with signal2:

    deteriorating = (
        filtered[
            "cadence_trend_status"
        ]
        .isin(
            [
                "Deteriorating",
                "Strongly Deteriorating",
            ]
        )
        .sum()
    )

    st.metric(
        "Cadence Deteriorating",
        f"{deteriorating:,.0f}",
    )

    st.caption(
        "Customers whose purchase intervals "
        "are progressively lengthening."
    )


with signal3:

    st.metric(
        "Priority Customer Margin",
        f"${priority_margin / 1_000_000:.2f}m",
    )

    st.caption(
        "Trailing 12-month margin associated "
        "with Protect Now and Proactive Retention customers."
    )


# =========================================================
# FOOTNOTE
# =========================================================

st.caption(
    "Active customer is currently defined as a purchase "
    "within the last 365 days. Behavioural lapse is assessed "
    "separately using each customer's observed purchase cadence."
)