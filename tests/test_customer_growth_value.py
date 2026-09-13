from pathlib import Path

import numpy as np
import pandas as pd


RUNTIME_DIR = Path("data/runtime")
GENERATED_DIR = Path("data/generated")

CUSTOMER_LTV_FILE = RUNTIME_DIR / "customer_ltv.parquet"
COHORT_RETENTION_FILE = RUNTIME_DIR / "cohort_retention.parquet"
COHORT_SUMMARY_FILE = RUNTIME_DIR / "cohort_summary.parquet"
CUSTOMER_GROWTH_MONTHLY_FILE = RUNTIME_DIR / "customer_growth_monthly.parquet"
TRANSACTION_FILE = RUNTIME_DIR / "golden_customer_transactions.parquet"
GOLDEN_MASTER_FILE = RUNTIME_DIR / "golden_customer_master.parquet"


def test_customer_ltv_reconciles_to_transactions():
    ltv = pd.read_parquet(CUSTOMER_LTV_FILE)
    transactions = pd.read_parquet(TRANSACTION_FILE)
    golden_master = pd.read_parquet(GOLDEN_MASTER_FILE)

    assert len(ltv) == len(golden_master)
    assert ltv["golden_customer_id"].is_unique
    assert set(ltv["golden_customer_id"]) == set(golden_master["golden_customer_id"])

    assert np.isclose(
        ltv["observed_ltv_sales"].sum(),
        transactions["net_sales"].sum(),
        atol=0.01,
    )

    assert np.isclose(
        ltv["observed_ltv_margin"].sum(),
        transactions["gross_margin"].sum(),
        atol=0.01,
    )


def test_ltv_maturity_is_monotonic_and_immature_values_are_blank():
    ltv = pd.read_parquet(CUSTOMER_LTV_FILE)

    windows = [3, 6, 12, 18, 24]

    mature_counts = [
        int(ltv[f"m{months}_mature"].sum())
        for months in windows
    ]

    assert mature_counts == sorted(
        mature_counts,
        reverse=True,
    )

    for months in windows:
        immature = ~ltv[f"m{months}_mature"]

        assert ltv.loc[
            immature,
            f"m{months}_margin",
        ].isna().all()

        assert ltv.loc[
            immature,
            f"m{months}_sales",
        ].isna().all()


def test_cohort_retention_starts_at_100_percent():
    retention = pd.read_parquet(COHORT_RETENTION_FILE)

    m0 = retention.loc[
        retention["cohort_age_month"].eq(0)
    ].copy()

    assert not m0.empty
    assert np.allclose(
        m0["retention_rate"],
        1.0,
    )

    assert (
        m0["active_customers"]
        == m0["cohort_customers"]
    ).all()


def test_customer_growth_monthly_reconciles():
    growth = pd.read_parquet(
        CUSTOMER_GROWTH_MONTHLY_FILE
    )

    reconciliation = (
        growth["opening_behaviourally_active"]
        + growth["new_customers"]
        + growth["reactivated_customers"]
        - growth["newly_lapsed_customers"]
    )

    assert (
        reconciliation
        == growth["closing_behaviourally_active"]
    ).all()

    assert (
        growth["closing_behaviourally_active"]
        >= 0
    ).all()


def test_cohort_summary_maturity_fields_are_consistent():
    summary = pd.read_parquet(COHORT_SUMMARY_FILE)

    for months in [3, 6, 12, 18, 24]:
        mature = summary[f"m{months}_mature"]

        assert summary.loc[
            ~mature,
            f"m{months}_margin_per_customer",
        ].isna().all()

        assert summary.loc[
            ~mature,
            f"m{months}_retention_rate",
        ].isna().all()
