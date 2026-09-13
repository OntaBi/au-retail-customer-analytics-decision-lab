from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.app.filters import render_customer_filters
from src.customer.customer_growth_value import (
    AS_OF_DATE,
    build_cohort_retention,
    build_cohort_summary,
    build_customer_growth_monthly,
    build_customer_movement_events,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "golden_customer_transactions.parquet"
)

CUSTOMER_LTV_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "customer_ltv.parquet"
)

CLUSTER_FILE = (
    APP_ROOT
    / "data"
    / "runtime"
    / "customer_clusters.parquet"
)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

@st.cache_data
def load_data():

    customer_ltv = pd.read_parquet(
        CUSTOMER_LTV_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    clusters = pd.read_parquet(
        CLUSTER_FILE
    )

    cluster_lookup = (
        clusters[
            [
                "golden_customer_id",
                "cluster_name",
            ]
        ]
        .drop_duplicates(
            subset=["golden_customer_id"]
        )
    )

    customer_ltv = customer_ltv.merge(
        cluster_lookup,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
    )

    customer_ltv[
        "cluster_name"
    ] = (
        customer_ltv[
            "cluster_name"
        ]
        .fillna("Not Yet Clustered")
    )

    return (
        customer_ltv,
        transactions,
    )


@st.cache_data
def build_filtered_growth_outputs(
    filtered_customer_ids: tuple,
    customer_ltv: pd.DataFrame,
    transactions: pd.DataFrame,
):

    selected_ids = set(
        filtered_customer_ids
    )

    selected_ltv = (
        customer_ltv.loc[
            customer_ltv[
                "golden_customer_id"
            ].isin(selected_ids)
        ]
        .copy()
    )

    selected_transactions = (
        transactions.loc[
            transactions[
                "golden_customer_id"
            ].isin(selected_ids)
        ]
        .copy()
    )

    if (
        selected_ltv.empty
        or selected_transactions.empty
    ):

        return (
            selected_ltv,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    cohort_retention = (
        build_cohort_retention(
            selected_ltv,
            selected_transactions,
        )
    )

    movement_events = (
        build_customer_movement_events(
            selected_transactions
        )
    )

    if movement_events.empty:

        customer_growth_monthly = (
            pd.DataFrame()
        )

    else:

        customer_growth_monthly = (
            build_customer_growth_monthly(
                movement_events,
                selected_transactions,
            )
        )

    cohort_summary = (
        build_cohort_summary(
            selected_ltv,
            cohort_retention,
        )
    )

    return (
        selected_ltv,
        cohort_retention,
        cohort_summary,
        customer_growth_monthly,
    )


@st.cache_data
def build_customer_value_funnel(
    filtered_customer_ids: tuple,
    customer_ltv: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    selected_ids = set(
        filtered_customer_ids
    )

    selected_ltv = (
        customer_ltv.loc[
            customer_ltv[
                "golden_customer_id"
            ].isin(selected_ids)
            & customer_ltv[
                "has_purchased"
            ]
            .fillna(False)
        ]
        .copy()
    )

    selected_transactions = (
        transactions.loc[
            transactions[
                "golden_customer_id"
            ].isin(selected_ids)
        ]
        .copy()
    )

    if (
        selected_ltv.empty
        or selected_transactions.empty
    ):

        return pd.DataFrame(
            columns=[
                "stage",
                "customers",
                "conversion_from_prior",
            ]
        )

    maturity_cutoff = (
        AS_OF_DATE
        - pd.DateOffset(
            months=12
        )
    )

    mature = (
        selected_ltv.loc[
            selected_ltv[
                "first_purchase_date"
            ].notna()
            & selected_ltv[
                "first_purchase_date"
            ].le(
                maturity_cutoff
            )
        ]
        .copy()
    )

    if mature.empty:

        return pd.DataFrame(
            columns=[
                "stage",
                "customers",
                "conversion_from_prior",
            ]
        )

    tx = (
        selected_transactions[
            [
                "golden_customer_id",
                "transaction_date",
                "order_id",
            ]
        ]
        .drop_duplicates()
        .merge(
            mature[
                [
                    "golden_customer_id",
                    "first_purchase_date",
                ]
            ],
            on="golden_customer_id",
            how="inner",
            validate="many_to_one",
        )
    )

    tx[
        "cohort_age_month"
    ] = (
        (
            tx[
                "transaction_date"
            ].dt.year
            * 12
            + tx[
                "transaction_date"
            ].dt.month
        )
        - (
            tx[
                "first_purchase_date"
            ].dt.year
            * 12
            + tx[
                "first_purchase_date"
            ].dt.month
        )
    )

    first_12m = (
        tx.loc[
            tx[
                "cohort_age_month"
            ].between(
                0,
                11,
            )
        ]
        .groupby(
            "golden_customer_id"
        )
        .agg(
            orders_first_12m=(
                "order_id",
                "nunique",
            ),
            active_m6_plus=(
                "cohort_age_month",
                lambda values:
                    values.between(
                        6,
                        11,
                    ).any(),
            ),
        )
        .reset_index()
    )

    m12_plus = (
        tx.loc[
            tx[
                "cohort_age_month"
            ].ge(12)
        ]
        .groupby(
            "golden_customer_id"
        )
        .agg(
            active_m12_plus=(
                "order_id",
                "nunique",
            )
        )
        .reset_index()
    )

    mature = (
        mature
        .merge(
            first_12m,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            m12_plus,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    mature[
        "orders_first_12m"
    ] = (
        mature[
            "orders_first_12m"
        ]
        .fillna(0)
    )

    mature[
        "active_m6_plus"
    ] = (
        mature[
            "active_m6_plus"
        ]
        .fillna(False)
    )

    mature[
        "active_m12_plus"
    ] = (
        mature[
            "active_m12_plus"
        ]
        .fillna(0)
        .gt(0)
    )

    acquired_mask = pd.Series(
        True,
        index=mature.index,
    )

    engaged_mask = (
        acquired_mask
        & mature[
            "orders_first_12m"
        ]
        .ge(3)
    )

    m6_mask = (
        engaged_mask
        & mature[
            "active_m6_plus"
        ]
    )

    m12_mask = (
        m6_mask
        & mature[
            "active_m12_plus"
        ]
    )

    high_value_mask = (
        m12_mask
        & mature[
            "customer_value_tier"
        ]
        .isin(
            [
                "High",
                "Very High",
            ]
        )
    )

    stages = [
        (
            "12M Mature Customers",
            int(
                acquired_mask.sum()
            ),
        ),
        (
            "Meaningfully Engaged",
            int(
                engaged_mask.sum()
            ),
        ),
        (
            "Retained at M6+",
            int(
                m6_mask.sum()
            ),
        ),
        (
            "Retained at M12+",
            int(
                m12_mask.sum()
            ),
        ),
        (
            "High-Value Customers",
            int(
                high_value_mask.sum()
            ),
        ),
    ]

    funnel = pd.DataFrame(
        stages,
        columns=[
            "stage",
            "customers",
        ],
    )

    funnel[
        "conversion_from_prior"
    ] = (
        funnel[
            "customers"
        ]
        / funnel[
            "customers"
        ]
        .shift(1)
    )

    funnel.loc[
        0,
        "conversion_from_prior",
    ] = 1.0

    return funnel


customer_ltv, transactions = load_data()

filtered_customers, filters = (
    render_customer_filters(
        customer_ltv
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
    "Customer Growth & Value"
)

st.caption(
    "How are customer acquisition, repeat behaviour, cohort retention "
    "and customer value evolving over time?"
)


# ---------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------

if filtered_customers.empty:

    st.warning(
        "No customers match the current filters."
    )

    st.stop()


selected_ids = tuple(
    sorted(
        filtered_customers[
            "golden_customer_id"
        ]
        .astype(str)
        .tolist()
    )
)

(
    growth_ltv,
    cohort_retention,
    cohort_summary,
    growth_monthly,
) = build_filtered_growth_outputs(
    selected_ids,
    customer_ltv,
    transactions,
)

customer_value_funnel = (
    build_customer_value_funnel(
        selected_ids,
        customer_ltv,
        transactions,
    )
)


# ---------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------

purchasers = int(
    growth_ltv[
        "has_purchased"
    ].sum()
)

repeat_customers = int(
    growth_ltv[
        "repeat_customer"
    ].sum()
)

repeat_rate = (
    repeat_customers
    / purchasers
    if purchasers > 0
    else 0
)

m12_mature = (
    growth_ltv[
        "m12_mature"
    ]
    .fillna(False)
)

m12_margin_ltv = (
    growth_ltv.loc[
        m12_mature,
        "m12_margin",
    ]
    .mean()
)

if pd.isna(m12_margin_ltv):
    m12_margin_ltv = 0.0

behaviourally_active = (
    int(
        growth_monthly[
            "closing_behaviourally_active"
        ].iloc[-1]
    )
    if not growth_monthly.empty
    else 0
)

latest_month = (
    growth_monthly[
        "month"
    ].max()
    if not growth_monthly.empty
    else pd.NaT
)

latest_net_movement = (
    int(
        growth_monthly.loc[
            growth_monthly[
                "month"
            ].eq(latest_month),
            "net_customer_movement",
        ].sum()
    )
    if pd.notna(latest_month)
    else 0
)


# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------

st.subheader(
    "Customer growth position"
)

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)

kpi1.metric(
    "Purchasing Customers",
    f"{purchasers:,}",
)

kpi2.metric(
    "Repeat Purchase Rate",
    f"{repeat_rate:.1%}",
)

kpi3.metric(
    "M12 Margin / Customer",
    f"${m12_margin_ltv:,.0f}",
)

kpi4.metric(
    "Behaviourally Active",
    f"{behaviourally_active:,}",
)

kpi5.metric(
    "Latest Net Movement",
    f"{latest_net_movement:+,}",
)

st.caption(
    "Repeat Purchase Rate = customers with 2+ observed orders among "
    "customers who have purchased. M12 Margin / Customer includes only "
    "customers with a complete 12-month observation window."
)


# ---------------------------------------------------------------------
# Customer movement
# ---------------------------------------------------------------------

st.subheader(
    "Customer movement"
)

st.caption(
    "Monthly customer inflows and outflows based on first purchase, "
    "behavioural lapse and subsequent reactivation."
)

if growth_monthly.empty:

    st.info(
        "Insufficient transaction history to calculate customer movement."
    )

else:

    movement_long = (
        growth_monthly[
            [
                "month",
                "new_customers",
                "reactivated_customers",
                "newly_lapsed_customers",
            ]
        ]
        .melt(
            id_vars="month",
            var_name="movement_type",
            value_name="customers",
        )
    )

    movement_label_map = {
        "new_customers":
            "New",
        "reactivated_customers":
            "Reactivated",
        "newly_lapsed_customers":
            "Newly Lapsed",
    }

    movement_long[
        "movement_type"
    ] = (
        movement_long[
            "movement_type"
        ]
        .map(
            movement_label_map
        )
    )

    movement_chart = px.bar(
        movement_long,
        x="month",
        y="customers",
        color="movement_type",
        barmode="group",
        labels={
            "month":
                "Month",
            "customers":
                "Customers",
            "movement_type":
                "Movement",
        },
    )

    movement_chart.add_trace(
        go.Scatter(
            x=growth_monthly[
                "month"
            ],
            y=growth_monthly[
                "net_customer_movement"
            ],
            mode="lines+markers",
            name="Net Movement",
            yaxis="y2",
        )
    )

    movement_chart.update_layout(
        xaxis_title=None,
        yaxis_title="Customer Events",
        yaxis2=dict(
            title="Net Movement",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=85,
        ),
    )

    st.plotly_chart(
        movement_chart,
        use_container_width=True,
    )

    recent_growth = (
        growth_monthly
        .tail(12)
        .copy()
    )

    movement_table = (
        recent_growth.rename(
            columns={
                "month":
                    "Month",
                "opening_behaviourally_active":
                    "Opening Active",
                "new_customers":
                    "New",
                "reactivated_customers":
                    "Reactivated",
                "newly_lapsed_customers":
                    "Newly Lapsed",
                "net_customer_movement":
                    "Net Movement",
                "closing_behaviourally_active":
                    "Closing Active",
            }
        )
    )

    movement_table[
        "Month"
    ] = (
        movement_table[
            "Month"
        ]
        .dt.strftime(
            "%b %Y"
        )
    )

    with st.expander(
        "View recent customer movement table"
    ):

        st.dataframe(
            movement_table,
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Customer value funnel
# ---------------------------------------------------------------------

st.subheader(
    "Customer value funnel"
)

st.caption(
    "Progression through repeat purchase, longer-term retention and "
    "commercial value using only customers with a complete 12-month "
    "observation window."
)

if customer_value_funnel.empty:

    st.info(
        "Insufficient mature customer history to calculate the value funnel."
    )

else:

    funnel_chart = go.Figure(
        go.Funnel(
            y=customer_value_funnel[
                "stage"
            ],
            x=customer_value_funnel[
                "customers"
            ],
            texttemplate=(
                "%{value:,} customers"
                "<br>%{percentPrevious:.1%} from prior stage"
            ),
            hovertemplate=(
                "%{y}"
                "<br>Customers: %{x:,}"
                "<br>Conversion from prior stage: "
                "%{percentPrevious:.1%}"
                "<extra></extra>"
            ),
        )
    )

    funnel_chart.update_layout(
        height=460,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        funnel_chart,
        use_container_width=True,
    )

    funnel_metrics = (
        customer_value_funnel.copy()
    )

    funnel_metrics[
        "conversion_from_prior"
    ] = (
        funnel_metrics[
            "conversion_from_prior"
        ]
        .map(
            lambda value:
                f"{value:.1%}"
        )
    )

    funnel_metrics = (
        funnel_metrics.rename(
            columns={
                "stage":
                    "Customer Stage",
                "customers":
                    "Customers",
                "conversion_from_prior":
                    "Conversion from Prior Stage",
            }
        )
    )

    with st.expander(
        "View customer value funnel detail"
    ):

        st.dataframe(
            funnel_metrics,
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------
# Cohort retention heatmap
# ---------------------------------------------------------------------

st.subheader(
    "Acquisition cohort retention"
)

st.caption(
    "Monthly purchasing activity by months since first purchase. "
    "Recent cohorts intentionally show fewer mature periods."
)

if cohort_retention.empty:

    st.info(
        "No cohort retention data is available for the selected customers."
    )

else:

    max_heatmap_age = min(
        12,
        int(
            cohort_retention[
                "cohort_age_month"
            ].max()
        ),
    )

    available_cohorts = (
        cohort_retention[
            "purchase_cohort_month"
        ]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    display_cohorts = set(
        available_cohorts.tail(12)
    )

    heatmap_data = (
        cohort_retention.loc[
            cohort_retention[
                "purchase_cohort_month"
            ].isin(display_cohorts)
            & cohort_retention[
                "cohort_age_month"
            ].between(
                0,
                max_heatmap_age,
            )
        ]
        .pivot(
            index="purchase_cohort_month",
            columns="cohort_age_month",
            values="retention_pct",
        )
        .sort_index(
            ascending=False
        )
    )

    heatmap_data.index = (
        heatmap_data.index.strftime(
            "%b %Y"
        )
    )

    heatmap_data.columns = [
        f"M{int(column)}"
        for column
        in heatmap_data.columns
    ]

    heatmap_text = (
        heatmap_data
        .apply(
            lambda column:
                column.map(
                    lambda value:
                        f"{value:.0f}%"
                        if pd.notna(value)
                        else ""
                )
        )
        .to_numpy()
    )

    retention_heatmap = px.imshow(
        heatmap_data,
        text_auto=False,
        aspect="auto",
        labels={
            "x":
                "Months Since First Purchase",
            "y":
                "Purchase Cohort",
            "color":
                "Retention %",
        },
    )

    retention_heatmap.update_traces(
        text=heatmap_text,
        texttemplate="%{text}",
        textfont=dict(
            size=12,
        ),
        hovertemplate=(
            "Cohort: %{y}"
            "<br>Month: %{x}"
            "<br>Retention: %{z:.1f}%"
            "<extra></extra>"
        ),
    )

    heatmap_height = (
        50 * len(heatmap_data.index)
        + 170
    )

    retention_heatmap.update_layout(
        height=heatmap_height,
        margin=dict(
            l=30,
            r=30,
            t=20,
            b=45,
        ),
        coloraxis_colorbar=dict(
            title="Retention %",
            len=0.78,
            y=0.5,
        ),
        xaxis=dict(
            side="bottom",
            tickangle=0,
        ),
        yaxis=dict(
            automargin=True,
        ),
    )

    st.plotly_chart(
        retention_heatmap,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Retention and repeat trends
# ---------------------------------------------------------------------

st.subheader(
    "Repeat purchase & retention trends"
)

trend_left, trend_right = (
    st.columns(2)
)

with trend_left:

    st.markdown(
        "**Repeat purchase rate by cohort**"
    )

    repeat_trend = (
        cohort_summary[
            [
                "purchase_cohort_month",
                "repeat_purchase_rate",
            ]
        ]
        .copy()
    )

    repeat_trend[
        "repeat_purchase_pct"
    ] = (
        repeat_trend[
            "repeat_purchase_rate"
        ]
        * 100
    )

    repeat_chart = px.line(
        repeat_trend,
        x="purchase_cohort_month",
        y="repeat_purchase_pct",
        markers=True,
        labels={
            "purchase_cohort_month":
                "Cohort",
            "repeat_purchase_pct":
                "Repeat Purchase Rate (%)",
        },
    )

    repeat_chart.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Repeat Purchase Rate (%)",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=80,
        ),
    )

    st.plotly_chart(
        repeat_chart,
        use_container_width=True,
    )

with trend_right:

    st.markdown(
        "**M6 retention by mature cohort**"
    )

    m6_trend = (
        cohort_summary.loc[
            cohort_summary[
                "m6_retention_rate"
            ].notna(),
            [
                "purchase_cohort_month",
                "m6_retention_rate",
                "m6_retention_prior6_benchmark",
            ],
        ]
        .copy()
    )

    m6_trend[
        "m6_retention_pct"
    ] = (
        m6_trend[
            "m6_retention_rate"
        ]
        * 100
    )

    m6_trend[
        "benchmark_pct"
    ] = (
        m6_trend[
            "m6_retention_prior6_benchmark"
        ]
        * 100
    )

    retention_chart = go.Figure()

    retention_chart.add_trace(
        go.Scatter(
            x=m6_trend[
                "purchase_cohort_month"
            ],
            y=m6_trend[
                "m6_retention_pct"
            ],
            mode="lines+markers",
            name="M6 Retention",
        )
    )

    retention_chart.add_trace(
        go.Scatter(
            x=m6_trend[
                "purchase_cohort_month"
            ],
            y=m6_trend[
                "benchmark_pct"
            ],
            mode="lines",
            name="Prior 6 Cohort Benchmark",
            line=dict(
                dash="dash",
            ),
        )
    )

    retention_chart.update_layout(
        xaxis_title=None,
        yaxis_title="M6 Retention (%)",
        legend=dict(
            title=None,
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=80,
        ),
    )

    st.plotly_chart(
        retention_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Cohort value development
# ---------------------------------------------------------------------

st.subheader(
    "Cohort value development"
)

st.caption(
    "Average cumulative customer margin within fixed observation windows. "
    "Only mature cohorts are included at each horizon."
)

value_columns = {
    "M3":
        "m3_margin_per_customer",
    "M6":
        "m6_margin_per_customer",
    "M12":
        "m12_margin_per_customer",
    "M18":
        "m18_margin_per_customer",
    "M24":
        "m24_margin_per_customer",
}

selected_value_horizon = (
    st.selectbox(
        "Value Horizon",
        options=list(
            value_columns.keys()
        ),
        index=2,
    )
)

selected_value_column = (
    value_columns[
        selected_value_horizon
    ]
)

cohort_value = (
    cohort_summary.loc[
        cohort_summary[
            selected_value_column
        ].notna(),
        [
            "purchase_cohort_month",
            selected_value_column,
        ],
    ]
    .copy()
)

if cohort_value.empty:

    st.info(
        "No mature cohorts are available for this value horizon."
    )

else:

    cohort_value_chart = px.line(
        cohort_value,
        x="purchase_cohort_month",
        y=selected_value_column,
        markers=True,
        labels={
            "purchase_cohort_month":
                "Purchase Cohort",
            selected_value_column:
                f"{selected_value_horizon} Margin / Customer",
        },
    )

    cohort_value_chart.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=(
            f"{selected_value_horizon} "
            "Margin / Customer"
        ),
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    cohort_value_chart.update_yaxes(
        tickprefix="$",
        separatethousands=True,
    )

    st.plotly_chart(
        cohort_value_chart,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Cohorts requiring attention
# ---------------------------------------------------------------------

st.subheader(
    "Cohorts requiring attention"
)

attention = (
    cohort_summary.loc[
        cohort_summary[
            "retention_trend"
        ]
        .isin(
            [
                "Deteriorating",
                "Improving",
            ]
        )
    ]
    .copy()
)

attention[
    "m6_retention_delta_pp"
] = (
    attention[
        "m6_retention_delta_vs_prior6"
    ]
    * 100
)

attention_display = (
    attention[
        [
            "purchase_cohort_month",
            "cohort_customers",
            "repeat_purchase_rate",
            "m6_retention_rate",
            "m6_retention_delta_pp",
            "m12_margin_per_customer",
            "retention_trend",
        ]
    ]
    .sort_values(
        [
            "retention_trend",
            "purchase_cohort_month",
        ]
    )
    .copy()
)

if attention_display.empty:

    st.info(
        "No mature cohorts currently exceed the ±5 percentage-point "
        "retention trend threshold."
    )

else:

    attention_display[
        "purchase_cohort_month"
    ] = (
        attention_display[
            "purchase_cohort_month"
        ]
        .dt.strftime(
            "%b %Y"
        )
    )

    attention_display[
        "repeat_purchase_rate"
    ] = (
        attention_display[
            "repeat_purchase_rate"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )

    attention_display[
        "m6_retention_rate"
    ] = (
        attention_display[
            "m6_retention_rate"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
                if pd.notna(x)
                else ""
        )
    )

    attention_display[
        "m6_retention_delta_pp"
    ] = (
        attention_display[
            "m6_retention_delta_pp"
        ]
        .map(
            lambda x:
                f"{x:+.1f}pp"
                if pd.notna(x)
                else ""
        )
    )

    attention_display[
        "m12_margin_per_customer"
    ] = (
        attention_display[
            "m12_margin_per_customer"
        ]
        .map(
            lambda x:
                f"${x:,.0f}"
                if pd.notna(x)
                else ""
        )
    )

    attention_display = (
        attention_display.rename(
            columns={
                "purchase_cohort_month":
                    "Cohort",
                "cohort_customers":
                    "Customers",
                "repeat_purchase_rate":
                    "Repeat Rate",
                "m6_retention_rate":
                    "M6 Retention",
                "m6_retention_delta_pp":
                    "vs Prior 6",
                "m12_margin_per_customer":
                    "M12 Margin / Customer",
                "retention_trend":
                    "Trend",
            }
        )
    )

    st.dataframe(
        attention_display,
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Executive decision signals
# ---------------------------------------------------------------------

st.subheader(
    "Executive decision signals"
)

mature_with_benchmark = (
    cohort_summary.loc[
        cohort_summary[
            "m6_retention_delta_vs_prior6"
        ].notna()
    ]
    .copy()
)

signal1, signal2, signal3 = (
    st.columns(3)
)

with signal1:

    st.markdown(
        "**Retention pressure**"
    )

    deteriorating = (
        mature_with_benchmark.loc[
            mature_with_benchmark[
                "m6_retention_delta_vs_prior6"
            ] <= -0.05
        ]
        .sort_values(
            "purchase_cohort_month",
            ascending=False,
        )
    )

    if deteriorating.empty:

        st.write(
            "No mature cohort is more than 5 percentage points "
            "below its prior-six-cohort M6 retention benchmark."
        )

    else:

        row = deteriorating.iloc[0]

        st.write(
            f"{row['purchase_cohort_month']:%b %Y} is "
            f"{abs(row['m6_retention_delta_vs_prior6']) * 100:.1f}pp "
            "below its prior-six-cohort M6 retention benchmark."
        )


with signal2:

    st.markdown(
        "**Customer value**"
    )

    mature_m12 = (
        cohort_summary.loc[
            cohort_summary[
                "m12_margin_per_customer"
            ].notna()
        ]
        .sort_values(
            "purchase_cohort_month"
        )
    )

    if len(mature_m12) < 2:

        st.write(
            "Insufficient mature cohorts for a 12-month value comparison."
        )

    else:

        latest_value = (
            mature_m12.iloc[-1]
        )

        prior_window = (
            mature_m12.iloc[
                max(
                    0,
                    len(mature_m12) - 7,
                ):
                -1
            ]
        )

        benchmark = (
            prior_window[
                "m12_margin_per_customer"
            ]
            .mean()
        )

        delta = (
            latest_value[
                "m12_margin_per_customer"
            ]
            / benchmark
            - 1
            if benchmark > 0
            else np.nan
        )

        if pd.isna(delta):

            st.write(
                "Insufficient benchmark history."
            )

        else:

            direction = (
                "above"
                if delta >= 0
                else "below"
            )

            st.write(
                f"{latest_value['purchase_cohort_month']:%b %Y} "
                f"M12 margin/customer is {abs(delta):.1%} {direction} "
                "the preceding mature-cohort benchmark."
            )


with signal3:

    st.markdown(
        "**Growth action**"
    )

    if growth_monthly.empty:

        st.write(
            "Insufficient customer movement history."
        )

    else:

        recent_3 = (
            growth_monthly
            .tail(3)
        )

        recent_net = int(
            recent_3[
                "net_customer_movement"
            ]
            .sum()
        )

        if recent_net < 0:

            st.write(
                f"Net behavioural customer movement is {recent_net:+,} "
                "across the latest three months. Prioritise retention "
                "and reactivation before increasing acquisition spend."
            )

        else:

            st.write(
                f"Net behavioural customer movement is {recent_net:+,} "
                "across the latest three months. Validate that growth "
                "is converting into repeat behaviour and customer margin."
            )


# ---------------------------------------------------------------------
# Full cohort performance table
# ---------------------------------------------------------------------

with st.expander(
    "View full cohort commercial performance"
):

    full_table = (
        cohort_summary[
            [
                "purchase_cohort_month",
                "cohort_customers",
                "repeat_purchase_rate",
                "observed_orders_per_customer",
                "observed_sales_per_customer",
                "observed_margin_per_customer",
                "m6_retention_rate",
                "m12_retention_rate",
                "m12_margin_per_customer",
                "retention_trend",
            ]
        ]
        .copy()
    )

    full_table[
        "purchase_cohort_month"
    ] = (
        full_table[
            "purchase_cohort_month"
        ]
        .dt.strftime(
            "%b %Y"
        )
    )

    full_table[
        "repeat_purchase_rate"
    ] = (
        full_table[
            "repeat_purchase_rate"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )

    for column in [
        "m6_retention_rate",
        "m12_retention_rate",
    ]:

        full_table[column] = (
            full_table[column]
            .map(
                lambda x:
                    f"{x:.1%}"
                    if pd.notna(x)
                    else ""
            )
        )

    full_table[
        "observed_orders_per_customer"
    ] = (
        full_table[
            "observed_orders_per_customer"
        ]
        .map(
            lambda x:
                f"{x:.1f}"
        )
    )

    for column in [
        "observed_sales_per_customer",
        "observed_margin_per_customer",
        "m12_margin_per_customer",
    ]:

        full_table[column] = (
            full_table[column]
            .map(
                lambda x:
                    f"${x:,.0f}"
                    if pd.notna(x)
                    else ""
            )
        )

    full_table = (
        full_table.rename(
            columns={
                "purchase_cohort_month":
                    "Cohort",
                "cohort_customers":
                    "Customers",
                "repeat_purchase_rate":
                    "Repeat Rate",
                "observed_orders_per_customer":
                    "Orders / Customer",
                "observed_sales_per_customer":
                    "Observed Sales / Customer",
                "observed_margin_per_customer":
                    "Observed Margin / Customer",
                "m6_retention_rate":
                    "M6 Retention",
                "m12_retention_rate":
                    "M12 Retention",
                "m12_margin_per_customer":
                    "M12 Margin / Customer",
                "retention_trend":
                    "Retention Trend",
            }
        )
    )

    st.dataframe(
        full_table,
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "Cohorts are based on first purchase month. Retention represents "
    "customers purchasing during each cohort-age month, not cumulative "
    "survival. Behavioural customer movement uses customer-specific "
    "purchase cadence to identify lapse and reactivation. Fixed-window "
    "customer value is reported only when the full observation window "
    "has matured."
)
