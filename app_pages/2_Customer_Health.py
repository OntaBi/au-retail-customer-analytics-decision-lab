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
    "Customer Health"
)

st.caption(
    "Which customers are behaving outside their normal "
    "purchase cadence, and where is deterioration emerging?"
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
# KPI CALCULATIONS
# =========================================================

total_customers = (
    filtered["golden_customer_id"]
    .nunique()
)

active_customers = int(
    filtered["active_customer"]
    .fillna(False)
    .sum()
)

watch_customers = int(
    filtered["lifecycle_status"]
    .eq("Watch")
    .sum()
)

at_risk_customers = int(
    filtered["lifecycle_status"]
    .eq("At Risk")
    .sum()
)

highly_lapsed_customers = int(
    filtered["lifecycle_status"]
    .eq("Highly Lapsed")
    .sum()
)

deteriorating_customers = int(
    filtered["cadence_trend_status"]
    .isin(
        [
            "Deteriorating",
            "Strongly Deteriorating",
        ]
    )
    .sum()
)


# =========================================================
# KPI ROW
# =========================================================

st.subheader(
    "Customer health position"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

with kpi1:
    st.metric(
        "Active Customers",
        f"{active_customers:,}",
    )
    st.caption(
        f"{active_customers / total_customers:.1%} of total customers"
    )

kpi2.metric(
    "Watch",
    f"{watch_customers:,}",
)

kpi3.metric(
    "At Risk",
    f"{at_risk_customers:,}",
)

kpi4.metric(
    "Highly Lapsed",
    f"{highly_lapsed_customers:,}",
)

kpi5.metric(
    "Cadence Deteriorating",
    f"{deteriorating_customers:,}",
)


# =========================================================
# LIFECYCLE + CADENCE TREND
# =========================================================

st.subheader(
    "Behavioural health signals"
)

left, right = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------

with left:

    st.markdown(
        "**Lifecycle status**"
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

    lifecycle_summary = (
        filtered[
            "lifecycle_status"
        ]
        .value_counts()
        .reindex(
            lifecycle_order,
            fill_value=0,
        )
        .rename_axis(
            "Lifecycle Status"
        )
        .reset_index(
            name="Customers"
        )
    )

    lifecycle_chart = px.bar(
        lifecycle_summary,
        x="Customers",
        y="Lifecycle Status",
        orientation="h",
    )

    lifecycle_chart.update_layout(
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

    lifecycle_chart.update_yaxes(
        categoryorder="array",
        categoryarray=lifecycle_order[::-1],
    )

    st.plotly_chart(
        lifecycle_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Cadence trend
# ---------------------------------------------------------

with right:

    st.markdown(
        "**Cadence trend**"
    )

    trend_order = [
        "Strongly Deteriorating",
        "Deteriorating",
        "Stable / Mixed",
        "Improving",
        "Insufficient History",
    ]

    trend_summary = (
        filtered[
            "cadence_trend_status"
        ]
        .value_counts()
        .reindex(
            trend_order,
            fill_value=0,
        )
        .rename_axis(
            "Cadence Trend"
        )
        .reset_index(
            name="Customers"
        )
    )

    trend_chart = px.bar(
        trend_summary,
        x="Customers",
        y="Cadence Trend",
        orientation="h",
    )

    trend_chart.update_layout(
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

    trend_chart.update_yaxes(
        categoryorder="array",
        categoryarray=trend_order[::-1],
    )

    st.plotly_chart(
        trend_chart,
        use_container_width=True,
    )


# =========================================================
# CADENCE EVIDENCE
# =========================================================

st.subheader(
    "Cadence evidence"
)

evidence_left, evidence_right = (
    st.columns([1, 1])
)


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------

with evidence_left:

    st.markdown(
        "**Cadence confidence**"
    )

    confidence_order = [
        "High",
        "Medium",
        "Low",
    ]

    confidence_summary = (
        filtered[
            "cadence_confidence"
        ]
        .value_counts()
        .reindex(
            confidence_order,
            fill_value=0,
        )
        .rename_axis(
            "Cadence Confidence"
        )
        .reset_index(
            name="Customers"
        )
    )

    confidence_chart = px.bar(
        confidence_summary,
        x="Cadence Confidence",
        y="Customers",
    )

    confidence_chart.update_layout(
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
        confidence_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Lapse ratio
# ---------------------------------------------------------

with evidence_right:

    st.markdown(
        "**Median lapse ratio by lifecycle status**"
    )

    lapse_summary = (
        filtered
        .groupby(
            "lifecycle_status",
            as_index=False,
        )
        .agg(
            median_lapse_ratio=(
                "adjusted_lapse_ratio",
                "median",
            )
        )
    )

    lapse_summary[
        "lifecycle_status"
    ] = pd.Categorical(
        lapse_summary[
            "lifecycle_status"
        ],
        categories=lifecycle_order,
        ordered=True,
    )

    lapse_summary = (
        lapse_summary
        .sort_values(
            "lifecycle_status"
        )
    )

    lapse_chart = px.bar(
        lapse_summary,
        x="median_lapse_ratio",
        y="lifecycle_status",
        orientation="h",
        labels={
            "lifecycle_status":
                "Lifecycle Status",
            "median_lapse_ratio":
                "Median Lapse Ratio",
        },
    )

    lapse_chart.add_vline(
        x=1.0,
        line_dash="dash",
        line_color="white",
        line_width=1.5,
        annotation_text="Expected cadence (1.0x)",
        annotation_position="top right",
        annotation_yshift=-2,
        annotation_font_color="white",
    )

    lapse_chart.update_layout(
        showlegend=False,
        xaxis_title="Median Lapse Ratio",
        yaxis_title=None,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    lapse_chart.update_yaxes(
        categoryorder="array",
        categoryarray=lifecycle_order[::-1],
    )

    st.plotly_chart(
        lapse_chart,
        use_container_width=True,
    )


# =========================================================
# HIGH-VALUE RISK MAP
# =========================================================

st.subheader(
    "High-value customer risk map"
)

st.caption(
    "High and Very High value customers positioned by "
    "behavioural lapse and trailing 12-month margin."
)

high_value = filtered.loc[
    filtered[
        "customer_value_tier"
    ].isin(
        [
            "High",
            "Very High",
        ]
    )
    & filtered[
        "adjusted_lapse_ratio"
    ].notna()
].copy()

if high_value.empty:

    st.info(
        "No High or Very High value customers "
        "with sufficient cadence history match the current filters."
    )

else:

    high_value["plot_sales"] = (
        high_value["trailing_12m_sales"]
        .fillna(0)
        .clip(lower=1)
    )

    risk_map = px.scatter(
        high_value,
        x="adjusted_lapse_ratio",
        y="trailing_12m_margin",
        size="plot_sales",
        color="cadence_trend_status",
        hover_name="golden_customer_id",
        hover_data={
            "cluster_name": True,
            "customer_value_tier": True,
            "lifecycle_status": True,
            "cadence_trend_status": True,
            "median_purchase_gap_days": ":.0f",
            "days_since_last_purchase": ":.0f",
            "trailing_12m_sales": ":,.0f",
            "trailing_12m_margin": ":,.0f",
            "decision_group": True,
            "plot_sales": False,
        },
        labels={
            "adjusted_lapse_ratio":
                "Lapse Ratio",
            "trailing_12m_margin":
                "Trailing 12M Margin",
            "cadence_trend_status":
                "Cadence Trend",
            "cluster_name":
                "Customer Segment",
            "customer_value_tier":
                "Value Tier",
            "lifecycle_status":
                "Lifecycle",
            "median_purchase_gap_days":
                "Typical Gap (Days)",
            "days_since_last_purchase":
                "Days Since Purchase",
            "trailing_12m_sales":
                "Trailing 12M Sales",
            "decision_group":
                "Decision",
        },
    )

    risk_map.add_vline(
        x=1.5,
        line_dash="dash",
        line_color="white",
        line_width=1.5,
        annotation_text="At-risk threshold (1.5x)",
        annotation_position="top right",
        annotation_yshift=-2,
        annotation_font_color="white",
    )

    risk_map.update_layout(
        xaxis_title="Lapse Ratio",
        yaxis_title="Trailing 12M Margin",
        legend_title="Cadence Trend",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        risk_map,
        use_container_width=True,
    )

    st.caption(
        "Customers further right are increasingly overdue relative "
        "to their own expected purchase cadence. Higher-positioned "
        "customers represent greater trailing margin value."
    )


# =========================================================
# PRIORITY RETENTION ACTION LIST
# =========================================================

st.subheader(
    "Priority Retention Action List"
)

st.caption(
    "High-priority customers for review or CRM activation based "
    "on customer value, lapse risk and deteriorating purchase behaviour."
)

watchlist_export = (
    filtered.loc[
        filtered[
            "cadence_trend_status"
        ].isin(
            [
                "Deteriorating",
                "Strongly Deteriorating",
            ]
        )
    ]
    .sort_values(
        [
            "priority_score",
            "gap_trend_pct",
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
            "median_purchase_gap_days",
            "days_since_last_purchase",
            "adjusted_lapse_ratio",
            "gap_trend_pct",
            "recent_vs_early_gap_pct",
            "trailing_12m_sales",
            "trailing_12m_margin",
            "priority_score",
            "decision_group",
            "recommended_action",
        ]
    ]
    .copy()
)

watchlist = (
    watchlist_export
    .head(20)
    .copy()
)


if watchlist.empty:

    st.info(
        "No deteriorating customers match "
        "the current filters."
    )

else:

    watchlist[
        "median_purchase_gap_days"
    ] = (
        watchlist[
            "median_purchase_gap_days"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    watchlist[
        "days_since_last_purchase"
    ] = (
        watchlist[
            "days_since_last_purchase"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    watchlist[
        "adjusted_lapse_ratio"
    ] = (
        watchlist[
            "adjusted_lapse_ratio"
        ]
        .map(
            lambda x:
                f"{x:.2f}x"
                if pd.notna(x)
                else ""
        )
    )

    watchlist[
        "gap_trend_pct"
    ] = (
        watchlist[
            "gap_trend_pct"
        ]
        .map(
            lambda x:
                f"{x:+.1%}"
                if pd.notna(x)
                else ""
        )
    )

    watchlist[
        "recent_vs_early_gap_pct"
    ] = (
        watchlist[
            "recent_vs_early_gap_pct"
        ]
        .map(
            lambda x:
                f"{x:+.1%}"
                if pd.notna(x)
                else ""
        )
    )

    watchlist[
        "trailing_12m_sales"
    ] = (
        watchlist[
            "trailing_12m_sales"
        ]
        .fillna(0)
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    watchlist[
        "trailing_12m_margin"
    ] = (
        watchlist[
            "trailing_12m_margin"
        ]
        .fillna(0)
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    watchlist[
        "priority_score"
    ] = (
        watchlist[
            "priority_score"
        ]
        .map(
            lambda x: f"{x:.1f}"
        )
    )

    watchlist = (
        watchlist.rename(
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
                "median_purchase_gap_days":
                    "Typical Gap (Days)",
                "days_since_last_purchase":
                    "Days Since Purchase",
                "adjusted_lapse_ratio":
                    "Lapse Ratio",
                "gap_trend_pct":
                    "Trend per Purchase",
                "recent_vs_early_gap_pct":
                    "Recent vs Early Gap",
                "trailing_12m_sales":
                    "Trailing 12M Sales",
                "trailing_12m_margin":
                    "Trailing 12M Margin",
                "priority_score":
                    "Priority Score",
                "decision_group":
                    "Decision",
                "recommended_action":
                    "Recommended Action",
            }
        )
    )

    st.dataframe(
        watchlist,
        hide_index=True,
        use_container_width=True,
    )

    crm_export = (
        watchlist_export.rename(
            columns={
                "golden_customer_id":
                    "golden_customer_id",
                "cluster_name":
                    "customer_segment",
                "customer_value_tier":
                    "value_tier",
                "lifecycle_status":
                    "lifecycle_status",
                "cadence_trend_status":
                    "cadence_trend",
                "median_purchase_gap_days":
                    "typical_gap_days",
                "days_since_last_purchase":
                    "days_since_purchase",
                "adjusted_lapse_ratio":
                    "lapse_ratio",
                "gap_trend_pct":
                    "gap_trend_pct",
                "recent_vs_early_gap_pct":
                    "recent_vs_early_gap_pct",
                "trailing_12m_sales":
                    "trailing_12m_sales",
                "trailing_12m_margin":
                    "trailing_12m_margin",
                "priority_score":
                    "priority_score",
                "decision_group":
                    "decision_group",
                "recommended_action":
                    "recommended_action",
            }
        )
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )

    st.download_button(
        label="Download CRM Action List",
        data=crm_export,
        file_name="customer_retention_action_list.csv",
        mime="text/csv",
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Active customer status uses a 365-day business rule. "
    "Behavioural lapse is assessed independently against each "
    "customer's expected purchase cadence. Cadence trend identifies "
    "customers whose purchase intervals are progressively lengthening, "
    "including customers who may still be active and on cadence today."
)
