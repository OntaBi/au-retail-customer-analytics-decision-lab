from pathlib import Path

import streamlit as st


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="AU Retail Customer Analytics Decision Lab",
    layout="wide",
)


# =========================================================
# PATHS
# =========================================================

APP_ROOT = Path(__file__).resolve().parent


# =========================================================
# PAGE NAVIGATION
# =========================================================

pages = [
    st.Page(
        APP_ROOT
        / "app_pages"
        / "1_Executive_Overview.py",
        title="Executive Overview",
        default=True,
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "2_Customer_Health.py",
        title="Customer Health",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "3_Customer_Segments.py",
        title="Customer Segments",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "4_Customer_Explorer.py",
        title="Customer Explorer",
    ),
    st.Page(
        APP_ROOT
        / "app_pages"
        / "5_Decision_Queue.py",
        title="Decision Queue",
    ),
]


# =========================================================
# TOP NAVIGATION
# =========================================================

navigation = st.navigation(
    pages,
    position="top",
)

navigation.run()