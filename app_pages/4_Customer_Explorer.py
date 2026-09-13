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

TRANSACTION_FILE = (APP_ROOT / "data" / "runtime" / "golden_customer_transactions.parquet")
IDENTITY_RESOLUTION_FILE = (APP_ROOT / "data" / "runtime" / "customer_identity_resolution.parquet")
IDENTITY_SOURCE_FILE = (APP_ROOT / "data" / "generated" / "customer_identity_records.parquet")


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

    transactions = pd.read_parquet(TRANSACTION_FILE)
    identity_resolution = pd.read_parquet(IDENTITY_RESOLUTION_FILE)
    identity_source = pd.read_parquet(IDENTITY_SOURCE_FILE)

    cluster_columns = [
        "golden_customer_id",
        "cluster",
        "cluster_name",
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

    customers = priority.merge(
        cluster_lookup,
        on="golden_customer_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_cluster",
        ),
    )

    customers["cluster_name"] = (
        customers["cluster_name"]
        .fillna("Not Yet Clustered")
    )

    identity_lookup = identity_resolution.merge(
        identity_source,
        on="identity_record_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )

    return customers, transactions, identity_lookup


customers, transactions, identity_lookup = load_data()


# =========================================================
# STANDARD FILTERS
# =========================================================

filtered_customers, filters = (
    render_customer_filters(customers)
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
    "Customer Explorer"
)

st.caption(
    "Explore an individual customer's value, shopping behaviour, "
    "purchase cadence and recommended action."
)


# =========================================================
# EMPTY FILTER CHECK
# =========================================================

if filtered_customers.empty:

    st.warning(
        "No customers match the current filters."
    )

    st.stop()


# =========================================================
# CUSTOMER SELECTOR
# =========================================================

st.subheader(
    "Select Golden Customer"
)

customer_options = (
    filtered_customers[
        "golden_customer_id"
    ]
    .sort_values()
    .tolist()
)

priority_candidates = (
    filtered_customers.loc[
        filtered_customers[
            "decision_group"
        ].isin(
            [
                "Protect Now",
                "Proactive Retention",
            ]
        )
    ]
    .sort_values(
        "priority_score",
        ascending=False,
    )[
        "golden_customer_id"
    ]
    .tolist()
)

default_customer = (
    priority_candidates[0]
    if priority_candidates
    else customer_options[0]
)

default_index = (
    customer_options.index(
        default_customer
    )
)

selector_col, search_col = st.columns(
    [2, 1]
)

with search_col:

    search_text = st.text_input(
        "Search Golden Customer ID",
        placeholder="e.g. GOLDEN_001234",
    )

if search_text:

    matching_customers = [
        customer_id
        for customer_id in customer_options
        if search_text.lower()
        in customer_id.lower()
    ]

else:

    matching_customers = customer_options


with selector_col:

    if not matching_customers:

        st.warning(
            "No customer IDs match the search."
        )

        st.stop()

    if (
        search_text
        or default_customer
        not in matching_customers
    ):
        selector_index = 0
    else:
        selector_index = (
            matching_customers.index(
                default_customer
            )
        )

    selected_customer = st.selectbox(
        "Golden Customer",
        options=matching_customers,
        index=selector_index,
    )


customer = filtered_customers.loc[
    filtered_customers["golden_customer_id"]
    == selected_customer
].iloc[0]

customer_transactions = (
    transactions.loc[
        transactions["golden_customer_id"]
        == selected_customer
    ]
    .sort_values(
        "transaction_date"
    )
    .copy()
)


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.subheader(
    "Customer profile"
)

profile1, profile2, profile3, profile4 = (
    st.columns(4)
)

with profile1:
    st.markdown(
        "**Customer Segment**"
    )
    st.write(
        customer[
            "cluster_name"
        ]
    )

with profile2:
    st.markdown(
        "**Value Tier**"
    )
    st.write(
        customer[
            "customer_value_tier"
        ]
    )

with profile3:
    st.markdown(
        "**Lifecycle**"
    )
    st.write(
        customer[
            "lifecycle_status"
        ]
    )

with profile4:
    st.markdown(
        "**Decision**"
    )
    st.write(
        customer[
            "decision_group"
        ]
    )


priority_col, action_col = st.columns(
    [1, 3]
)

with priority_col:
    st.metric(
        "Priority Score",
        f"{customer['priority_score']:.1f}",
    )

with action_col:
    st.markdown(
        "**Recommended Action**"
    )
    st.info(
        customer[
            "recommended_action"
        ]
    )


# =========================================================
# IDENTITY & DATA CONFIDENCE
# =========================================================
st.subheader("Identity & data confidence")

customer_identity_records = identity_lookup.loc[
    identity_lookup["golden_customer_id"] == selected_customer
].copy()

if customer_identity_records.empty:
    st.info(
        "No source identity records are linked to this Golden Customer."
    )
else:
    source_records_linked = len(customer_identity_records)
    source_systems = (
        customer_identity_records["source_system"]
        .dropna()
        .nunique()
    )
    best_match_row = (
        customer_identity_records
        .sort_values("match_confidence", ascending=False)
        .iloc[0]
    )
    minimum_confidence = (
        customer_identity_records["match_confidence"].min()
    )
    resolution_status = (
        "Resolved Multi-Record"
        if source_records_linked > 1
        else "Standalone"
    )

    id1, id2, id3, id4, id5 = st.columns(5)
    with id1:
        st.markdown("**Golden Customer ID**")
        st.write(selected_customer)
    id2.metric("Source Records Linked", f"{source_records_linked:,.0f}")
    id3.metric("Source Systems", f"{source_systems:,.0f}")
    id4.metric("Minimum Match Confidence", f"{minimum_confidence:.1%}")
    with id5:
        st.markdown("**Resolution Status**")
        st.write(resolution_status)

    st.caption(
        f"Strongest recorded match method: {best_match_row['match_method']}. "
        "Identity evidence is shown from observable source records only; hidden synthetic "
        "ground truth is reserved for QA and is not used by the application."
    )

    identity_display_columns = [
        column
        for column in [
            "identity_record_id",
            "source_system",
            "source_role",
            "match_method",
            "match_confidence",
            "ambiguous_match_flag",
        ]
        if column in customer_identity_records.columns
    ]

    if identity_display_columns:
        identity_display = (
            customer_identity_records[identity_display_columns]
            .sort_values(
                ["source_system", "identity_record_id"],
                na_position="last",
            )
            .copy()
        )

        if "match_confidence" in identity_display.columns:
            identity_display["match_confidence"] = (
                identity_display["match_confidence"]
                .map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
            )

        identity_display = identity_display.rename(
            columns={
                "identity_record_id": "Source Identity Record",
                "source_system": "Source System",
                "source_role": "Source Role",
                "match_method": "Match Method",
                "match_confidence": "Match Confidence",
                "ambiguous_match_flag": "Ambiguous Match",
            }
        )

        with st.expander("View linked source identity records"):
            st.dataframe(
                identity_display,
                hide_index=True,
                use_container_width=True,
            )


# =========================================================
# COMMERCIAL VALUE
# =========================================================

st.subheader(
    "Commercial value"
)

value1, value2, value3, value4, value5 = (
    st.columns(5)
)

trailing_sales = (
    customer[
        "trailing_12m_sales"
    ]
    if pd.notna(
        customer[
            "trailing_12m_sales"
        ]
    )
    else 0
)

trailing_margin = (
    customer[
        "trailing_12m_margin"
    ]
    if pd.notna(
        customer[
            "trailing_12m_margin"
        ]
    )
    else 0
)

lifetime_orders = (
    customer[
        "orders"
    ]
    if pd.notna(
        customer["orders"]
    )
    else 0
)

avg_order_value = (
    customer[
        "avg_order_value"
    ]
    if pd.notna(
        customer[
            "avg_order_value"
        ]
    )
    else 0
)

margin_per_order = (
    customer[
        "avg_margin_per_order"
    ]
    if "avg_margin_per_order"
    in customer.index
    and pd.notna(
        customer[
            "avg_margin_per_order"
        ]
    )
    else 0
)

value1.metric(
    "Trailing 12M Sales",
    f"${trailing_sales:,.0f}",
)

value2.metric(
    "Trailing 12M Margin",
    f"${trailing_margin:,.0f}",
)

value3.metric(
    "Lifetime Orders",
    f"{lifetime_orders:,.0f}",
)

value4.metric(
    "Average Order Value",
    f"${avg_order_value:,.0f}",
)

value5.metric(
    "Margin / Order",
    f"${margin_per_order:,.0f}",
)


# =========================================================
# CUSTOMER HEALTH
# =========================================================

st.subheader(
    "Customer health"
)

health1, health2, health3, health4, health5 = (
    st.columns(5)
)

days_since_purchase = (
    customer[
        "days_since_last_purchase"
    ]
    if pd.notna(
        customer[
            "days_since_last_purchase"
        ]
    )
    else None
)

typical_gap = (
    customer[
        "median_purchase_gap_days"
    ]
    if pd.notna(
        customer[
            "median_purchase_gap_days"
        ]
    )
    else None
)

lapse_ratio = (
    customer[
        "adjusted_lapse_ratio"
    ]
    if pd.notna(
        customer[
            "adjusted_lapse_ratio"
        ]
    )
    else None
)

health1.metric(
    "Days Since Purchase",
    (
        f"{days_since_purchase:,.0f}"
        if days_since_purchase
        is not None
        else "N/A"
    ),
)

health2.metric(
    "Typical Purchase Gap",
    (
        f"{typical_gap:,.0f} days"
        if typical_gap
        is not None
        else "N/A"
    ),
)

health3.metric(
    "Lapse Ratio",
    (
        f"{lapse_ratio:.2f}x"
        if lapse_ratio
        is not None
        else "N/A"
    ),
)

with health4:
    st.markdown(
        "**Cadence Confidence**"
    )
    st.write(
        customer[
            "cadence_confidence"
        ]
    )

with health5:
    st.markdown(
        "**Cadence Trend**"
    )
    st.write(
        customer[
            "cadence_trend_status"
        ]
    )


# =========================================================
# DECISION SIGNALS
# =========================================================

st.subheader(
    "Decision signals"
)

reason_col1, reason_col2, reason_col3 = (
    st.columns(3)
)

with reason_col1:

    st.markdown(
        "**Commercial value signal**"
    )

    st.metric(
        "Value Score",
        f"{customer['value_score']:.1f} / 100",
    )

    st.caption(
        "Based on trailing sales and margin contribution."
    )


with reason_col2:

    st.markdown(
        "**Lapse risk signal**"
    )

    st.metric(
        "Lapse Risk Score",
        f"{customer['lapse_risk_score']:.1f} / 100",
    )

    st.caption(
        "Based on how far the customer is beyond "
        "their expected purchase cadence."
    )


with reason_col3:

    st.markdown(
        "**Momentum risk signal**"
    )

    st.metric(
        "Momentum Risk Score",
        f"{customer['momentum_risk_score']:.1f} / 100",
    )

    st.caption(
        "Based on longitudinal deterioration in "
        "purchase cadence."
    )


# =========================================================
# PURCHASE HISTORY
# =========================================================

st.subheader(
    "Purchase history"
)

if customer_transactions.empty:

    st.info(
        "This customer has no recorded purchases."
    )

else:

    monthly_history = (
        customer_transactions
        .assign(
            month=lambda x:
                x[
                    "transaction_date"
                ].dt.to_period(
                    "M"
                )
        )
        .groupby(
            "month",
            as_index=False,
        )
        .agg(
            orders=(
                "order_id",
                "nunique",
            ),
            sales=(
                "net_sales",
                "sum",
            ),
            margin=(
                "gross_margin",
                "sum",
            ),
        )
    )

    monthly_history[
        "month_label"
    ] = (
        monthly_history[
            "month"
        ]
        .astype(str)
        .map(
            lambda value:
                pd.Period(
                    value,
                    freq="M",
                ).strftime(
                    "%b %Y"
                )
        )
    )

    history_chart = px.bar(
        monthly_history,
        x="month_label",
        y="sales",
        hover_data={
            "orders": ":,",
            "margin": ":,.0f",
        },
        labels={
            "month_label":
                "Month",
            "sales":
                "Sales",
            "orders":
                "Orders",
            "margin":
                "Margin",
        },
    )

    history_chart.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Sales",
        bargap=0.35,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        history_chart,
        use_container_width=True,
    )


# =========================================================
# PURCHASE CADENCE
# =========================================================

st.subheader(
    "Purchase cadence"
)

if len(
    customer_transactions
) < 2:

    st.info(
        "Insufficient purchase history to calculate cadence."
    )

else:

    purchase_dates = (
        customer_transactions[
            [
                "transaction_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "transaction_date"
        )
        .copy()
    )

    purchase_dates[
        "previous_purchase_date"
    ] = (
        purchase_dates[
            "transaction_date"
        ]
        .shift(1)
    )

    purchase_dates[
        "purchase_gap_days"
    ] = (
        purchase_dates[
            "transaction_date"
        ]
        - purchase_dates[
            "previous_purchase_date"
        ]
    ).dt.days

    purchase_dates = (
        purchase_dates
        .dropna(
            subset=[
                "purchase_gap_days",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    purchase_dates[
        "purchase_sequence"
    ] = (
        purchase_dates.index
        + 1
    )

    cadence_chart = px.line(
        purchase_dates,
        x="purchase_sequence",
        y="purchase_gap_days",
        markers=True,
        labels={
            "purchase_sequence":
                "Purchase Gap Sequence",
            "purchase_gap_days":
                "Days Between Purchases",
        },
    )

    if typical_gap is not None:

        cadence_chart.add_hline(
            y=typical_gap,
            line_dash="dash",
            line_color="white",
            annotation_text=(
                f"Typical gap "
                f"({typical_gap:.0f} days)"
            ),
            annotation_position="top right",
            annotation_yshift=-18,
            annotation_font_color="white",
        )

    cadence_chart.update_layout(
        showlegend=False,
        xaxis_title="Purchase Gap Sequence",
        yaxis_title="Days Between Purchases",
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),
    )

    st.plotly_chart(
        cadence_chart,
        use_container_width=True,
    )


# =========================================================
# SHOPPING BEHAVIOUR
# =========================================================

st.subheader(
    "Shopping behaviour"
)

behaviour_left, behaviour_right = (
    st.columns([1, 1])
)


# ---------------------------------------------------------
# Channel mix
# ---------------------------------------------------------

with behaviour_left:

    st.markdown(
        "**Channel mix**"
    )

    if customer_transactions.empty:

        st.info(
            "No channel history available."
        )

    else:

        channel_mix = (
            customer_transactions[
                "channel"
            ]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .rename_axis(
                "Channel"
            )
            .reset_index(
                name="Share"
            )
            .sort_values(
                "Share",
                ascending=True,
            )
        )

        channel_chart = px.bar(
            channel_mix,
            x="Share",
            y="Channel",
            orientation="h",
            text="Share",
        )

        channel_chart.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside",
            cliponaxis=False,
        )

        channel_chart.update_layout(
            showlegend=False,
            height=250,
            xaxis_title="Order Share (%)",
            yaxis_title=None,
            xaxis_range=[
                0,
                110,
            ],
            margin=dict(
                l=20,
                r=50,
                t=10,
                b=20,
            ),
        )

        st.plotly_chart(
            channel_chart,
            use_container_width=True,
        )


# ---------------------------------------------------------
# Category mix
# ---------------------------------------------------------

with behaviour_right:

    st.markdown(
        "**Category mix**"
    )

    if customer_transactions.empty:

        st.info(
            "No category history available."
        )

    else:

        category_mix = (
            customer_transactions[
                "category"
            ]
            .value_counts(
                normalize=True
            )
            .mul(100)
            .rename_axis(
                "Category"
            )
            .reset_index(
                name="Share"
            )
            .sort_values(
                "Share",
                ascending=True,
            )
        )

        category_chart = px.bar(
            category_mix,
            x="Share",
            y="Category",
            orientation="h",
            text="Share",
        )

        category_chart.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside",
            cliponaxis=False,
        )

        category_chart.update_layout(
            showlegend=False,
            height=250,
            xaxis_title="Order Share (%)",
            yaxis_title=None,
            xaxis_range=[
                0,
                110,
            ],
            margin=dict(
                l=20,
                r=50,
                t=10,
                b=20,
            ),
        )

        st.plotly_chart(
            category_chart,
            use_container_width=True,
        )


# =========================================================
# PROMOTION BEHAVIOUR
# =========================================================

st.subheader(
    "Promotion behaviour"
)

promo1, promo2, promo3, promo4 = (
    st.columns(4)
)

avg_discount = (
    customer[
        "avg_discount_pct"
    ]
    if "avg_discount_pct"
    in customer.index
    and pd.notna(
        customer[
            "avg_discount_pct"
        ]
    )
    else 0
)

discount_share = (
    customer[
        "discounted_order_share"
    ]
    if "discounted_order_share"
    in customer.index
    and pd.notna(
        customer[
            "discounted_order_share"
        ]
    )
    else 0
)

channels_used = (
    customer[
        "channels_used"
    ]
    if "channels_used"
    in customer.index
    and pd.notna(
        customer[
            "channels_used"
        ]
    )
    else 0
)

categories_used = (
    customer[
        "categories_used"
    ]
    if "categories_used"
    in customer.index
    and pd.notna(
        customer[
            "categories_used"
        ]
    )
    else 0
)

promo1.metric(
    "Average Discount",
    f"{avg_discount:.1%}",
)

promo2.metric(
    "Discounted Orders",
    f"{discount_share:.1%}",
)

promo3.metric(
    "Channels Used",
    f"{channels_used:,.0f}",
)

promo4.metric(
    "Categories Used",
    f"{categories_used:,.0f}",
)


# =========================================================
# RECENT TRANSACTIONS
# =========================================================

st.subheader(
    "Recent transactions"
)

if customer_transactions.empty:

    st.info(
        "No transactions available."
    )

else:

    recent_transactions = (
        customer_transactions
        .sort_values(
            "transaction_date",
            ascending=False,
        )
        .head(20)
        [
            [
                "transaction_date",
                "order_id",
                "channel",
                "category",
                "units",
                "discount_pct",
                "net_sales",
                "gross_margin",
            ]
        ]
        .copy()
    )

    recent_transactions[
        "transaction_date"
    ] = (
        recent_transactions[
            "transaction_date"
        ]
        .dt.strftime(
            "%d %b %Y"
        )
    )

    recent_transactions[
        "discount_pct"
    ] = (
        recent_transactions[
            "discount_pct"
        ]
        .map(
            lambda x: f"{x:.0%}"
        )
    )

    recent_transactions[
        "net_sales"
    ] = (
        recent_transactions[
            "net_sales"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    recent_transactions[
        "gross_margin"
    ] = (
        recent_transactions[
            "gross_margin"
        ]
        .map(
            lambda x: f"${x:,.2f}"
        )
    )

    recent_transactions = (
        recent_transactions.rename(
            columns={
                "transaction_date":
                    "Date",
                "order_id":
                    "Order",
                "channel":
                    "Channel",
                "category":
                    "Category",
                "units":
                    "Units",
                "discount_pct":
                    "Discount",
                "net_sales":
                    "Sales",
                "gross_margin":
                    "Margin",
            }
        )
    )

    st.dataframe(
        recent_transactions,
        hide_index=True,
        use_container_width=True,
    )


# =========================================================
# CUSTOMER EXPORT
# =========================================================

st.subheader(
    "Golden Customer record export"
)

export_columns = [
    "golden_customer_id",
    "cluster_name",
    "customer_value_tier",
    "lifecycle_status",
    "cadence_trend_status",
    "days_since_last_purchase",
    "median_purchase_gap_days",
    "adjusted_lapse_ratio",
    "trailing_12m_sales",
    "trailing_12m_margin",
    "priority_score",
    "decision_group",
    "recommended_action",
]

customer_export = (
    customers.loc[
        customers["golden_customer_id"]
        == selected_customer,
        export_columns,
    ]
    .rename(
        columns={
            "cluster_name":
                "customer_segment",
            "customer_value_tier":
                "value_tier",
            "cadence_trend_status":
                "cadence_trend",
            "days_since_last_purchase":
                "days_since_purchase",
            "median_purchase_gap_days":
                "typical_gap_days",
            "adjusted_lapse_ratio":
                "lapse_ratio",
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
    label="Download Golden Customer Record",
    data=customer_export,
    file_name=(
        f"{selected_customer}_golden_customer_record.csv"
    ),
    mime="text/csv",
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Customer Explorer combines identity evidence, commercial value, behavioural "
    "segmentation, purchase cadence and decision-engine outputs "
    "for an individual Golden Customer. Behavioural lapse is assessed "
    "relative to the customer's own observed purchasing pattern "
    "rather than a universal inactivity threshold."
)
