import pandas as pd
import streamlit as st


def render_customer_filters(
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    filtered = customers.copy()

    st.sidebar.header(
        "Filters"
    )

    # ---------------------------------------------------------
    # State
    # ---------------------------------------------------------

    states = [
        "All",
        *sorted(
            filtered["state"]
            .dropna()
            .unique()
            .tolist()
        ),
    ]

    selected_state = st.sidebar.selectbox(
        "State",
        options=states,
    )

    if selected_state != "All":
        filtered = filtered.loc[
            filtered["state"]
            == selected_state
        ]

    # ---------------------------------------------------------
    # Behavioural segment
    # ---------------------------------------------------------

    segments = [
        "All",
        *sorted(
            filtered["cluster_name"]
            .dropna()
            .unique()
            .tolist()
        ),
    ]

    selected_segment = st.sidebar.selectbox(
        "Customer Segment",
        options=segments,
    )

    if selected_segment != "All":
        filtered = filtered.loc[
            filtered["cluster_name"]
            == selected_segment
        ]

    # ---------------------------------------------------------
    # Value tier
    # ---------------------------------------------------------

    value_order = [
        "Very High",
        "High",
        "Medium",
        "Low",
        "No Purchase",
    ]

    available_value_tiers = (
        filtered["customer_value_tier"]
        .dropna()
        .unique()
        .tolist()
    )

    value_tiers = [
        "All",
        *[
            value
            for value in value_order
            if value in available_value_tiers
        ],
    ]

    selected_value_tier = (
        st.sidebar.selectbox(
            "Value Tier",
            options=value_tiers,
        )
    )

    if selected_value_tier != "All":
        filtered = filtered.loc[
            filtered["customer_value_tier"]
            == selected_value_tier
        ]

    filters = {
        "state": selected_state,
        "segment": selected_segment,
        "value_tier": selected_value_tier,
    }

    return filtered, filters