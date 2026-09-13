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

    cluster_columns = [
        "golden_customer_id",
        "cluster",
        "cluster_name",
        "pca_1",
        "pca_2",
        "orders",
        "avg_order_value",
        "avg_margin_per_order",
        "units_per_order",
        "avg_discount_pct",
        "discounted_order_share",
        "store_share",
        "online_share",
        "click_collect_share",
        "channels_used",
        "categories_used",
        "category_concentration",
        "dominant_category_share",
        "median_purchase_gap_days",
        "cadence_cv",
    ]

    cluster_lookup = clusters[
        [
            column
            for column in cluster_columns
            if column in clusters.columns
        ]
    ].copy()

    data = priority.merge(
        cluster_lookup,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_cluster",
        ),
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
    "Customer Segments"
)

st.caption(
    "How do behavioural customer groups differ in value, "
    "shopping behaviour and commercial opportunity?"
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
# SEGMENT KPI CALCULATIONS
# =========================================================

clustered = filtered.loc[
    filtered["cluster_name"]
    != "Not Yet Clustered"
].copy()

total_customers = (
    filtered["golden_customer_id"]
    .nunique()
)

clustered_customers = (
    clustered["golden_customer_id"]
    .nunique()
)

segment_count = (
    clustered["cluster_name"]
    .nunique()
)

top_segment = (
    clustered["cluster_name"]
    .value_counts()
    .index[0]
    if not clustered.empty
    else "N/A"
)

very_high_value = int(
    filtered[
        "customer_value_tier"
    ]
    .eq("Very High")
    .sum()
)

priority_customers = int(
    filtered[
        "decision_group"
    ]
    .isin(
        [
            "Protect Now",
            "Proactive Retention",
        ]
    )
    .sum()
)


# =========================================================
# KPI ROW
# =========================================================

st.subheader(
    "Segment position"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

with kpi1:
    st.metric(
        "Customers",
        f"{total_customers:,}",
    )

with kpi2:
    st.metric(
        "Behaviourally Clustered",
        f"{clustered_customers:,}",
    )

    st.caption(
        f"{clustered_customers / total_customers:.1%} of customers"
    )

with kpi3:
    st.metric(
        "Customer Segments",
        f"{segment_count:,}",
    )

with kpi4:
    st.metric(
        "Very High Value",
        f"{very_high_value:,}",
    )

with kpi5:
    st.metric(
        "Protect / Retain",
        f"{priority_customers:,}",
    )


# =========================================================
# SEGMENT DISTRIBUTION + VALUE
# =========================================================

st.subheader(
    "Customer segment mix"
)

left, right = st.columns(
    [1, 1]
)


# ---------------------------------------------------------
# Customer share
# ---------------------------------------------------------

with left:

    st.markdown(
        "**Customers by segment**"
    )

    segment_summary = (
        filtered
        .groupby(
            "cluster_name",
            as_index=False,
        )
        .agg(
            customers=(
                "golden_customer_id",
                "nunique",
            ),
        )
    )

    segment_summary[
        "customer_share"
    ] = (
        segment_summary[
            "customers"
        ]
        / segment_summary[
            "customers"
        ].sum()
    )

    segment_summary = (
        segment_summary
        .sort_values(
            "customers",
            ascending=True,
        )
    )

    segment_chart = px.bar(
        segment_summary,
        x="customers",
        y="cluster_name",
        orientation="h",
        hover_data={
            "customer_share":
                ":.1%",
        },
        labels={
            "cluster_name":
                "Customer Segment",
            "customers":
                "Customers",
            "customer_share":
                "Customer Share",
        },
    )

    segment_chart.update_layout(
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
        segment_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Margin by segment
# ---------------------------------------------------------

with right:

    st.markdown(
        "**Trailing 12M margin by segment**"
    )

    margin_summary = (
        filtered
        .groupby(
            "cluster_name",
            as_index=False,
        )
        .agg(
            trailing_12m_margin=(
                "trailing_12m_margin",
                "sum",
            ),
        )
        .sort_values(
            "trailing_12m_margin",
            ascending=True,
        )
    )

    margin_chart = px.bar(
        margin_summary,
        x="trailing_12m_margin",
        y="cluster_name",
        orientation="h",
        labels={
            "cluster_name":
                "Customer Segment",
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

    st.plotly_chart(
        margin_chart,
        use_container_width=True,
    )


# =========================================================
# SEGMENT BEHAVIOURAL MAP
# =========================================================

st.subheader(
    "Behavioural segment map"
)

st.caption(
    "Two-dimensional PCA projection of the behavioural "
    "features used by the clustering model."
)

pca_data = filtered.loc[
    filtered["pca_1"].notna()
    & filtered["pca_2"].notna()
    & (
        filtered["cluster_name"]
        != "Not Yet Clustered"
    )
].copy()

if pca_data.empty:

    st.info(
        "No clustered customers match the current filters."
    )

else:

    pca_chart = px.scatter(
        pca_data,
        x="pca_1",
        y="pca_2",
        color="cluster_name",
        hover_name="golden_customer_id",
        hover_data={
            "cluster_name": True,
            "customer_value_tier": True,
            "lifecycle_status": True,
            "decision_group": True,
            "trailing_12m_sales": ":,.0f",
            "trailing_12m_margin": ":,.0f",
            "pca_1": False,
            "pca_2": False,
        },
        labels={
            "pca_1":
                "Behavioural Dimension 1",
            "pca_2":
                "Behavioural Dimension 2",
            "cluster_name":
                "Customer Segment",
            "customer_value_tier":
                "Value Tier",
            "lifecycle_status":
                "Lifecycle",
            "decision_group":
                "Decision",
            "trailing_12m_sales":
                "Trailing 12M Sales",
            "trailing_12m_margin":
                "Trailing 12M Margin",
        },
        opacity=0.55,
    )

    pca_chart.update_layout(
        xaxis_title="Behavioural Dimension 1",
        yaxis_title="Behavioural Dimension 2",
        legend_title="Customer Segment",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        pca_chart,
        use_container_width=True,
    )

    st.caption(
        "PCA is used for visualisation only. "
        "Customer clusters were fitted using the full "
        "standardised behavioural feature set."
    )


# =========================================================
# SEGMENT BEHAVIOURAL PROFILE
# =========================================================

st.subheader(
    "Segment behavioural profile"
)

profile_source = filtered.loc[
    filtered["cluster_name"]
    != "Not Yet Clustered"
].copy()

if profile_source.empty:

    st.info(
        "No clustered customers match the current filters."
    )

else:

    segment_profile = (
        profile_source
        .groupby(
            "cluster_name",
            as_index=False,
        )
        .agg(
            customers=(
                "golden_customer_id",
                "nunique",
            ),
            median_orders=(
                "orders",
                "median",
            ),
            median_aov=(
                "avg_order_value",
                "median",
            ),
            median_margin_per_order=(
                "avg_margin_per_order",
                "median",
            ),
            median_discount_pct=(
                "avg_discount_pct",
                "median",
            ),
            median_store_share=(
                "store_share",
                "median",
            ),
            median_online_share=(
                "online_share",
                "median",
            ),
            median_click_collect_share=(
                "click_collect_share",
                "median",
            ),
            median_categories_used=(
                "categories_used",
                "median",
            ),
            median_category_concentration=(
                "category_concentration",
                "median",
            ),
            median_purchase_gap=(
                "median_purchase_gap_days",
                "median",
            ),
        )
    )

    segment_profile[
        "customer_share"
    ] = (
        segment_profile[
            "customers"
        ]
        / segment_profile[
            "customers"
        ].sum()
    )

    display_profile = (
        segment_profile.copy()
    )

    display_profile[
        "customer_share"
    ] = (
        display_profile[
            "customer_share"
        ]
        .map(
            lambda x: f"{x:.1%}"
        )
    )

    display_profile[
        "median_orders"
    ] = (
        display_profile[
            "median_orders"
        ]
        .map(
            lambda x: f"{x:,.0f}"
        )
    )

    display_profile[
        "median_aov"
    ] = (
        display_profile[
            "median_aov"
        ]
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    display_profile[
        "median_margin_per_order"
    ] = (
        display_profile[
            "median_margin_per_order"
        ]
        .map(
            lambda x: f"${x:,.0f}"
        )
    )

    for column in [
        "median_discount_pct",
        "median_store_share",
        "median_online_share",
        "median_click_collect_share",
        "median_category_concentration",
    ]:

        display_profile[column] = (
            display_profile[
                column
            ]
            .map(
                lambda x:
                    f"{x:.1%}"
                    if pd.notna(x)
                    else ""
            )
        )

    display_profile[
        "median_categories_used"
    ] = (
        display_profile[
            "median_categories_used"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    display_profile[
        "median_purchase_gap"
    ] = (
        display_profile[
            "median_purchase_gap"
        ]
        .map(
            lambda x:
                f"{x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    display_profile = (
        display_profile.rename(
            columns={
                "cluster_name":
                    "Customer Segment",
                "customers":
                    "Customers",
                "customer_share":
                    "Share",
                "median_orders":
                    "Median Orders",
                "median_aov":
                    "Median AOV",
                "median_margin_per_order":
                    "Median Margin / Order",
                "median_discount_pct":
                    "Median Discount %",
                "median_store_share":
                    "Store Share",
                "median_online_share":
                    "Online Share",
                "median_click_collect_share":
                    "C&C Share",
                "median_categories_used":
                    "Categories Used",
                "median_category_concentration":
                    "Category Concentration",
                "median_purchase_gap":
                    "Typical Gap (Days)",
            }
        )
    )

    st.dataframe(
        display_profile,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# VALUE MIX BY SEGMENT
# =========================================================

st.subheader(
    "Customer value mix by segment"
)

value_order = [
    "Very High",
    "High",
    "Medium",
    "Low",
    "No Purchase",
]

value_mix = (
    filtered.loc[
        filtered[
            "cluster_name"
        ]
        != "Not Yet Clustered"
    ]
    .groupby(
        [
            "cluster_name",
            "customer_value_tier",
        ]
    )
    .size()
    .reset_index(
        name="customers"
    )
)

if value_mix.empty:

    st.info(
        "No clustered customers match the current filters."
    )

else:

    value_mix[
        "customer_value_tier"
    ] = pd.Categorical(
        value_mix[
            "customer_value_tier"
        ],
        categories=value_order,
        ordered=True,
    )

    segment_totals = (
        value_mix
        .groupby(
            "cluster_name"
        )["customers"]
        .transform("sum")
    )

    value_mix[
        "segment_share"
    ] = (
        value_mix[
            "customers"
        ]
        / segment_totals
    )

    value_chart = px.bar(
        value_mix,
        x="cluster_name",
        y="segment_share",
        color="customer_value_tier",
        category_orders={
            "customer_value_tier":
                value_order,
        },
        hover_data={
            "customers": ":,",
            "segment_share": ":.1%",
        },
        labels={
            "cluster_name":
                "Customer Segment",
            "segment_share":
                "Share of Segment",
            "customer_value_tier":
                "Value Tier",
            "customers":
                "Customers",
        },
    )

    value_chart.update_layout(
        barmode="stack",
        xaxis_title=None,
        yaxis_title="Share of Segment",
        yaxis_tickformat=".0%",
        legend_title="Value Tier",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        value_chart,
        use_container_width=True,
    )


# =========================================================
# DECISION MIX BY SEGMENT
# =========================================================

st.subheader(
    "Decision mix by segment"
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

decision_mix = (
    filtered.loc[
        filtered[
            "cluster_name"
        ]
        != "Not Yet Clustered"
    ]
    .groupby(
        [
            "cluster_name",
            "decision_group",
        ]
    )
    .size()
    .reset_index(
        name="customers"
    )
)

if decision_mix.empty:

    st.info(
        "No clustered customers match the current filters."
    )

else:

    decision_mix[
        "decision_group"
    ] = pd.Categorical(
        decision_mix[
            "decision_group"
        ],
        categories=decision_order,
        ordered=True,
    )

    segment_totals = (
        decision_mix
        .groupby(
            "cluster_name"
        )["customers"]
        .transform("sum")
    )

    decision_mix[
        "segment_share"
    ] = (
        decision_mix[
            "customers"
        ]
        / segment_totals
    )

    decision_chart = px.bar(
        decision_mix,
        x="cluster_name",
        y="segment_share",
        color="decision_group",
        category_orders={
            "decision_group":
                decision_order,
        },
        hover_data={
            "customers": ":,",
            "segment_share": ":.1%",
        },
        labels={
            "cluster_name":
                "Customer Segment",
            "segment_share":
                "Share of Segment",
            "decision_group":
                "Decision Group",
            "customers":
                "Customers",
        },
    )

    decision_chart.update_layout(
        barmode="stack",
        xaxis_title=None,
        yaxis_title="Share of Segment",
        yaxis_tickformat=".0%",
        legend_title="Decision Group",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        decision_chart,
        use_container_width=True,
    )


# =========================================================
# SEGMENT COMMERCIAL SUMMARY
# =========================================================

st.subheader(
    "Segment commercial summary"
)

commercial_summary = (
    filtered
    .groupby(
        "cluster_name",
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
        protect_now=(
            "decision_group",
            lambda x:
                (x == "Protect Now").sum(),
        ),
        proactive_retention=(
            "decision_group",
            lambda x:
                (
                    x
                    == "Proactive Retention"
                ).sum(),
        ),
    )
)

commercial_summary[
    "priority_customers"
] = (
    commercial_summary[
        "protect_now"
    ]
    + commercial_summary[
        "proactive_retention"
    ]
)

commercial_summary = (
    commercial_summary
    .sort_values(
        "trailing_12m_margin",
        ascending=False,
    )
)

display_summary = (
    commercial_summary.copy()
)

display_summary[
    "trailing_12m_sales"
] = (
    display_summary[
        "trailing_12m_sales"
    ]
    .fillna(0)
    .map(
        lambda x:
            f"${x / 1_000_000:.2f}m"
    )
)

display_summary[
    "trailing_12m_margin"
] = (
    display_summary[
        "trailing_12m_margin"
    ]
    .fillna(0)
    .map(
        lambda x:
            f"${x / 1_000_000:.2f}m"
    )
)

display_summary[
    "median_priority_score"
] = (
    display_summary[
        "median_priority_score"
    ]
    .map(
        lambda x:
            f"{x:.1f}"
            if pd.notna(x)
            else ""
    )
)

display_summary = (
    display_summary[
        [
            "cluster_name",
            "customers",
            "trailing_12m_sales",
            "trailing_12m_margin",
            "priority_customers",
            "median_priority_score",
        ]
    ]
    .rename(
        columns={
            "cluster_name":
                "Customer Segment",
            "customers":
                "Customers",
            "trailing_12m_sales":
                "Trailing 12M Sales",
            "trailing_12m_margin":
                "Trailing 12M Margin",
            "priority_customers":
                "Protect / Retain",
            "median_priority_score":
                "Median Priority Score",
        }
    )
)

st.dataframe(
    display_summary,
    hide_index=True,
    use_container_width=True,
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Behavioural segments are derived using K-Means clustering "
    "across purchase intensity, order value, promotion behaviour, "
    "channel mix, category breadth and purchase cadence. "
    "K=6 was selected using statistical diagnostics together with "
    "cluster stability and commercial interpretability. "
    "Synthetic persona and behavioural archetype labels were used "
    "for validation only and were not inputs to the clustering model."
)