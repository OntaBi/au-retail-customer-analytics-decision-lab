from pathlib import Path

import numpy as np
import pandas as pd


AS_OF_DATE = pd.Timestamp("2026-07-31")

CUSTOMER_MASTER_FILE = Path("data/generated/customer_master.parquet")
TRANSACTION_FILE = Path("data/generated/transactions.parquet")
CUSTOMER_VALUE_FILE = Path("data/runtime/customer_value.parquet")

OUTPUT_DIR = Path("data/runtime")
CUSTOMER_LTV_FILE = OUTPUT_DIR / "customer_ltv.parquet"
COHORT_RETENTION_FILE = OUTPUT_DIR / "cohort_retention.parquet"
COHORT_SUMMARY_FILE = OUTPUT_DIR / "cohort_summary.parquet"
CUSTOMER_GROWTH_MONTHLY_FILE = OUTPUT_DIR / "customer_growth_monthly.parquet"

LTV_WINDOWS_MONTHS = [3, 6, 12, 18, 24]

MIN_PRIOR_GAPS_FOR_BEHAVIOURAL_EVENT = 2
LAPSE_MULTIPLIER = 2.0
MIN_EXPECTED_RETURN_DAYS = 7


def month_start(series: pd.Series) -> pd.Series:
    return series.dt.to_period("M").dt.to_timestamp()


def month_difference(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later.dt.year * 12 + later.dt.month) - (
        earlier.dt.year * 12 + earlier.dt.month
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_customer_ltv(
    customer_master: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_value: pd.DataFrame,
) -> pd.DataFrame:
    ordered_transactions = transactions.sort_values(
        ["customer_id", "transaction_date", "order_id"]
    ).copy()

    base = (
        ordered_transactions.groupby("customer_id")
        .agg(
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
            observed_orders=("order_id", "nunique"),
            observed_units=("units", "sum"),
            observed_ltv_sales=("net_sales", "sum"),
            observed_ltv_margin=("gross_margin", "sum"),
        )
        .reset_index()
    )

    purchase_sequence = (
        ordered_transactions[
            ["customer_id", "transaction_date", "order_id"]
        ]
        .drop_duplicates()
        .sort_values(["customer_id", "transaction_date", "order_id"])
        .copy()
    )

    purchase_sequence["purchase_number"] = (
        purchase_sequence.groupby("customer_id").cumcount() + 1
    )

    second_purchase = (
        purchase_sequence.loc[
            purchase_sequence["purchase_number"].eq(2),
            ["customer_id", "transaction_date"],
        ]
        .rename(columns={"transaction_date": "second_purchase_date"})
    )

    customer_ltv = (
        customer_master[
            ["customer_id", "state", "acquisition_date"]
        ]
        .merge(base, on="customer_id", how="left", validate="one_to_one")
        .merge(
            second_purchase,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    value_columns = [
        "customer_id",
        "customer_value_tier",
        "rfm_segment",
        "commercial_value_score",
        "trailing_12m_sales",
        "trailing_12m_margin",
    ]

    available_value_columns = [
        column for column in value_columns if column in customer_value.columns
    ]

    customer_ltv = customer_ltv.merge(
        customer_value[available_value_columns],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    customer_ltv["acquisition_cohort_month"] = month_start(
        customer_ltv["acquisition_date"]
    )

    customer_ltv["purchase_cohort_month"] = month_start(
        customer_ltv["first_purchase_date"]
    )

    customer_ltv["customer_age_days"] = (
        AS_OF_DATE - customer_ltv["acquisition_date"]
    ).dt.days

    customer_ltv["purchase_tenure_days"] = (
        AS_OF_DATE - customer_ltv["first_purchase_date"]
    ).dt.days

    customer_ltv["acquisition_to_first_purchase_days"] = (
        customer_ltv["first_purchase_date"] - customer_ltv["acquisition_date"]
    ).dt.days

    customer_ltv["days_to_second_purchase"] = (
        customer_ltv["second_purchase_date"] - customer_ltv["first_purchase_date"]
    ).dt.days

    customer_ltv["has_purchased"] = (
        customer_ltv["observed_orders"].fillna(0).gt(0)
    )

    customer_ltv["repeat_customer"] = (
        customer_ltv["observed_orders"].fillna(0).ge(2)
    )

    for column in [
        "observed_orders",
        "observed_units",
        "observed_ltv_sales",
        "observed_ltv_margin",
    ]:
        customer_ltv[column] = customer_ltv[column].fillna(0)

    customer_ltv["observed_margin_rate"] = safe_divide(
        customer_ltv["observed_ltv_margin"],
        customer_ltv["observed_ltv_sales"],
    )

    transaction_value = (
        ordered_transactions[
            ["customer_id", "transaction_date", "net_sales", "gross_margin"]
        ]
        .merge(
            customer_ltv[["customer_id", "first_purchase_date"]],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    for months in LTV_WINDOWS_MONTHS:
        window_end = (
            transaction_value["first_purchase_date"]
            + pd.DateOffset(months=months)
        )

        in_window = transaction_value["transaction_date"] < window_end

        window_value = (
            transaction_value.loc[in_window]
            .groupby("customer_id")
            .agg(
                **{
                    f"m{months}_sales": ("net_sales", "sum"),
                    f"m{months}_margin": ("gross_margin", "sum"),
                }
            )
            .reset_index()
        )

        customer_ltv = customer_ltv.merge(
            window_value,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

        maturity_date = (
            customer_ltv["first_purchase_date"]
            + pd.DateOffset(months=months)
        )

        mature = (
            customer_ltv["first_purchase_date"].notna()
            & (maturity_date <= AS_OF_DATE)
        )

        customer_ltv[f"m{months}_mature"] = mature

        for metric in ["sales", "margin"]:
            column = f"m{months}_{metric}"
            customer_ltv[column] = customer_ltv[column].fillna(0)
            customer_ltv.loc[~mature, column] = np.nan

    return customer_ltv


def build_cohort_retention(
    customer_ltv: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    purchased_customers = customer_ltv.loc[
        customer_ltv["has_purchased"],
        ["customer_id", "first_purchase_date", "purchase_cohort_month"],
    ].copy()

    cohort_sizes = (
        purchased_customers.groupby("purchase_cohort_month")
        .agg(cohort_customers=("customer_id", "nunique"))
        .reset_index()
    )

    activity = (
        transactions[["customer_id", "transaction_date"]]
        .drop_duplicates()
        .merge(
            purchased_customers,
            on="customer_id",
            how="inner",
            validate="many_to_one",
        )
    )

    activity["activity_month"] = month_start(activity["transaction_date"])

    activity["cohort_age_month"] = month_difference(
        activity["activity_month"],
        activity["purchase_cohort_month"],
    )

    active_by_age = (
        activity.groupby(["purchase_cohort_month", "cohort_age_month"])
        .agg(active_customers=("customer_id", "nunique"))
        .reset_index()
    )

    grid_rows = []

    for row in cohort_sizes.itertuples(index=False):
        cohort_month = row.purchase_cohort_month

        cohort_max_age = (
            AS_OF_DATE.year * 12
            + AS_OF_DATE.month
            - (cohort_month.year * 12 + cohort_month.month)
        )

        for cohort_age_month in range(0, cohort_max_age + 1):
            grid_rows.append(
                {
                    "purchase_cohort_month": cohort_month,
                    "cohort_age_month": cohort_age_month,
                    "cohort_customers": row.cohort_customers,
                }
            )

    retention = pd.DataFrame(grid_rows)

    retention = retention.merge(
        active_by_age,
        on=["purchase_cohort_month", "cohort_age_month"],
        how="left",
        validate="one_to_one",
    )

    retention["active_customers"] = (
        retention["active_customers"].fillna(0).astype(int)
    )

    retention["retention_rate"] = (
        retention["active_customers"] / retention["cohort_customers"]
    )

    retention["retention_pct"] = (
        retention["retention_rate"] * 100
    ).round(1)

    return retention


def build_customer_movement_events(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    purchase_dates = (
        transactions[["customer_id", "transaction_date"]]
        .drop_duplicates()
        .sort_values(["customer_id", "transaction_date"])
    )

    events = []

    for customer_id, customer_history in purchase_dates.groupby(
        "customer_id",
        sort=False,
    ):
        dates = (
            customer_history["transaction_date"]
            .sort_values()
            .tolist()
        )

        if not dates:
            continue

        events.append(
            {
                "customer_id": customer_id,
                "event_date": dates[0],
                "event_type": "New",
                "active_change": 1,
            }
        )

        observed_gaps = []

        for index in range(1, len(dates)):
            previous_date = dates[index - 1]
            current_date = dates[index]

            current_gap = (current_date - previous_date).days

            if (
                len(observed_gaps)
                >= MIN_PRIOR_GAPS_FOR_BEHAVIOURAL_EVENT
            ):
                expected_gap = max(
                    MIN_EXPECTED_RETURN_DAYS,
                    float(np.median(observed_gaps)),
                )

                lapse_threshold_days = (
                    expected_gap * LAPSE_MULTIPLIER
                )

                if current_gap > lapse_threshold_days:
                    lapse_date = (
                        previous_date
                        + pd.Timedelta(
                            days=int(np.ceil(lapse_threshold_days))
                        )
                    )

                    if lapse_date <= AS_OF_DATE:
                        events.append(
                            {
                                "customer_id": customer_id,
                                "event_date": lapse_date,
                                "event_type": "Lapsed",
                                "active_change": -1,
                            }
                        )

                    if current_date <= AS_OF_DATE:
                        events.append(
                            {
                                "customer_id": customer_id,
                                "event_date": current_date,
                                "event_type": "Reactivated",
                                "active_change": 1,
                            }
                        )

            observed_gaps.append(current_gap)

        if (
            len(observed_gaps)
            >= MIN_PRIOR_GAPS_FOR_BEHAVIOURAL_EVENT
        ):
            expected_gap = max(
                MIN_EXPECTED_RETURN_DAYS,
                float(np.median(observed_gaps)),
            )

            lapse_threshold_days = (
                expected_gap * LAPSE_MULTIPLIER
            )

            final_lapse_date = (
                dates[-1]
                + pd.Timedelta(
                    days=int(np.ceil(lapse_threshold_days))
                )
            )

            if final_lapse_date <= AS_OF_DATE:
                events.append(
                    {
                        "customer_id": customer_id,
                        "event_date": final_lapse_date,
                        "event_type": "Lapsed",
                        "active_change": -1,
                    }
                )

    return pd.DataFrame(events)


def build_customer_growth_monthly(
    movement_events: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    movement_events = movement_events.copy()

    movement_events["month"] = month_start(
        movement_events["event_date"]
    )

    monthly_events = (
        movement_events
        .pivot_table(
            index="month",
            columns="event_type",
            values="customer_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
    )

    for event_type in ["New", "Reactivated", "Lapsed"]:
        if event_type not in monthly_events.columns:
            monthly_events[event_type] = 0

    start_month = (
        transactions["transaction_date"]
        .min()
        .to_period("M")
        .to_timestamp()
    )

    end_month = AS_OF_DATE.to_period("M").to_timestamp()

    full_months = pd.DataFrame(
        {
            "month": pd.date_range(
                start=start_month,
                end=end_month,
                freq="MS",
            )
        }
    )

    growth = (
        full_months
        .merge(monthly_events, on="month", how="left")
        .fillna(
            {
                "New": 0,
                "Reactivated": 0,
                "Lapsed": 0,
            }
        )
    )

    for column in ["New", "Reactivated", "Lapsed"]:
        growth[column] = growth[column].astype(int)

    growth = growth.rename(
        columns={
            "New": "new_customers",
            "Reactivated": "reactivated_customers",
            "Lapsed": "newly_lapsed_customers",
        }
    )

    growth["net_customer_movement"] = (
        growth["new_customers"]
        + growth["reactivated_customers"]
        - growth["newly_lapsed_customers"]
    )

    growth["closing_behaviourally_active"] = (
        growth["net_customer_movement"].cumsum()
    )

    growth["opening_behaviourally_active"] = (
        growth["closing_behaviourally_active"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    growth["closing_behaviourally_active"] = (
        growth["closing_behaviourally_active"].astype(int)
    )

    return growth[
        [
            "month",
            "opening_behaviourally_active",
            "new_customers",
            "reactivated_customers",
            "newly_lapsed_customers",
            "net_customer_movement",
            "closing_behaviourally_active",
        ]
    ]


def build_cohort_summary(
    customer_ltv: pd.DataFrame,
    cohort_retention: pd.DataFrame,
) -> pd.DataFrame:
    purchased = customer_ltv.loc[
        customer_ltv["has_purchased"]
    ].copy()

    summary = (
        purchased.groupby("purchase_cohort_month")
        .agg(
            cohort_customers=("customer_id", "nunique"),
            repeat_customers=("repeat_customer", "sum"),
            observed_orders=("observed_orders", "sum"),
            observed_sales=("observed_ltv_sales", "sum"),
            observed_margin=("observed_ltv_margin", "sum"),
        )
        .reset_index()
    )

    summary["repeat_purchase_rate"] = (
        summary["repeat_customers"] / summary["cohort_customers"]
    )

    summary["observed_orders_per_customer"] = (
        summary["observed_orders"] / summary["cohort_customers"]
    )

    summary["observed_sales_per_customer"] = (
        summary["observed_sales"] / summary["cohort_customers"]
    )

    summary["observed_margin_per_customer"] = (
        summary["observed_margin"] / summary["cohort_customers"]
    )

    for months in LTV_WINDOWS_MONTHS:
        mature_customer_values = (
            purchased.loc[purchased[f"m{months}_mature"]]
            .groupby("purchase_cohort_month")
            .agg(
                **{
                    f"m{months}_sales_per_customer": (
                        f"m{months}_sales",
                        "mean",
                    ),
                    f"m{months}_margin_per_customer": (
                        f"m{months}_margin",
                        "mean",
                    ),
                }
            )
            .reset_index()
        )

        summary = summary.merge(
            mature_customer_values,
            on="purchase_cohort_month",
            how="left",
            validate="one_to_one",
        )

        cohort_mature = (
            AS_OF_DATE
            >= (
                summary["purchase_cohort_month"]
                + pd.offsets.MonthEnd(0)
                + pd.DateOffset(months=months)
            )
        )

        summary[f"m{months}_mature"] = cohort_mature

        summary.loc[
            ~cohort_mature,
            [
                f"m{months}_sales_per_customer",
                f"m{months}_margin_per_customer",
            ],
        ] = np.nan

        retention_at_age = (
            cohort_retention.loc[
                cohort_retention["cohort_age_month"].eq(months),
                ["purchase_cohort_month", "retention_rate"],
            ]
            .rename(
                columns={
                    "retention_rate":
                        f"m{months}_retention_rate"
                }
            )
        )

        summary = summary.merge(
            retention_at_age,
            on="purchase_cohort_month",
            how="left",
            validate="one_to_one",
        )

        summary.loc[
            ~cohort_mature,
            f"m{months}_retention_rate",
        ] = np.nan

    summary = summary.sort_values(
        "purchase_cohort_month"
    ).reset_index(drop=True)

    mature_retention = (
        summary.loc[
            summary["m6_retention_rate"].notna(),
            ["purchase_cohort_month", "m6_retention_rate"],
        ]
        .copy()
    )

    mature_retention["m6_retention_prior6_benchmark"] = (
        mature_retention["m6_retention_rate"]
        .rolling(window=6, min_periods=3)
        .mean()
        .shift(1)
    )

    summary = summary.merge(
        mature_retention[
            [
                "purchase_cohort_month",
                "m6_retention_prior6_benchmark",
            ]
        ],
        on="purchase_cohort_month",
        how="left",
        validate="one_to_one",
    )

    summary["m6_retention_delta_vs_prior6"] = (
        summary["m6_retention_rate"]
        - summary["m6_retention_prior6_benchmark"]
    )

    summary["retention_trend"] = np.select(
        [
            summary["m6_retention_delta_vs_prior6"] <= -0.05,
            summary["m6_retention_delta_vs_prior6"] >= 0.05,
        ],
        [
            "Deteriorating",
            "Improving",
        ],
        default="Stable / Mixed",
    )

    summary.loc[
        summary["m6_retention_delta_vs_prior6"].isna(),
        "retention_trend",
    ] = "Insufficient History"

    return summary


def run_qa(
    customer_ltv: pd.DataFrame,
    cohort_retention: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    customer_growth_monthly: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    print("\nCUSTOMER GROWTH & VALUE QA")
    print("=" * 80)

    print(f"Customers: {len(customer_ltv):,}")
    print(
        "Customers with purchases: "
        f"{customer_ltv['has_purchased'].sum():,}"
    )
    print(
        "Repeat customers: "
        f"{customer_ltv['repeat_customer'].sum():,}"
    )

    repeat_rate = (
        customer_ltv.loc[
            customer_ltv["has_purchased"],
            "repeat_customer",
        ].mean()
    )

    print(
        f"Repeat purchase rate: "
        f"{repeat_rate:.1%}"
    )

    print(
        f"Observed customer sales: "
        f"${customer_ltv['observed_ltv_sales'].sum():,.0f}"
    )

    print(
        f"Observed customer margin: "
        f"${customer_ltv['observed_ltv_margin'].sum():,.0f}"
    )

    print("\nLTV maturity:")

    for months in LTV_WINDOWS_MONTHS:
        mature_customers = customer_ltv[
            f"m{months}_mature"
        ].sum()

        median_margin = (
            customer_ltv.loc[
                customer_ltv[f"m{months}_mature"],
                f"m{months}_margin",
            ].median()
        )

        print(
            f"M{months}: "
            f"{mature_customers:,} mature customers | "
            f"median margin ${median_margin:,.2f}"
        )

    print("\nCOHORT QA")
    print("-" * 80)

    print(
        f"Purchase cohorts: "
        f"{cohort_summary['purchase_cohort_month'].nunique():,}"
    )

    print(
        f"Cohort range: "
        f"{cohort_summary['purchase_cohort_month'].min().date()} "
        f"to "
        f"{cohort_summary['purchase_cohort_month'].max().date()}"
    )

    m0 = cohort_retention.loc[
        cohort_retention["cohort_age_month"].eq(0),
        "retention_rate",
    ]

    print(
        f"M0 retention range: "
        f"{m0.min():.1%} to {m0.max():.1%}"
    )

    print("\nRecent mature cohort summary:")

    display_columns = [
        "purchase_cohort_month",
        "cohort_customers",
        "repeat_purchase_rate",
        "m6_retention_rate",
        "m12_retention_rate",
        "m12_margin_per_customer",
        "retention_trend",
    ]

    print(
        cohort_summary[display_columns]
        .tail(12)
        .to_string(index=False)
    )

    print("\nCUSTOMER MOVEMENT QA")
    print("-" * 80)

    total_new = (
        customer_growth_monthly["new_customers"].sum()
    )

    total_reactivated = (
        customer_growth_monthly[
            "reactivated_customers"
        ].sum()
    )

    total_lapsed = (
        customer_growth_monthly[
            "newly_lapsed_customers"
        ].sum()
    )

    print(f"New customer events: {total_new:,}")
    print(f"Reactivation events: {total_reactivated:,}")
    print(f"Lapse events: {total_lapsed:,}")

    print(
        "Closing behaviourally active: "
        f"{customer_growth_monthly['closing_behaviourally_active'].iloc[-1]:,}"
    )

    reconciliation = (
        customer_growth_monthly[
            "opening_behaviourally_active"
        ]
        + customer_growth_monthly[
            "new_customers"
        ]
        + customer_growth_monthly[
            "reactivated_customers"
        ]
        - customer_growth_monthly[
            "newly_lapsed_customers"
        ]
        - customer_growth_monthly[
            "closing_behaviourally_active"
        ]
    )

    print(
        "Monthly movement reconciliation max abs difference: "
        f"{reconciliation.abs().max():,.0f}"
    )

    transaction_sales = transactions["net_sales"].sum()
    ltv_sales = customer_ltv["observed_ltv_sales"].sum()

    print(
        "Transaction-to-LTV sales reconciliation: "
        f"${transaction_sales - ltv_sales:,.2f}"
    )


def build_customer_growth_value(
    customer_master: pd.DataFrame,
    transactions: pd.DataFrame,
    customer_value: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    customer_ltv = build_customer_ltv(
        customer_master,
        transactions,
        customer_value,
    )

    cohort_retention = build_cohort_retention(
        customer_ltv,
        transactions,
    )

    movement_events = build_customer_movement_events(
        transactions
    )

    customer_growth_monthly = build_customer_growth_monthly(
        movement_events,
        transactions,
    )

    cohort_summary = build_cohort_summary(
        customer_ltv,
        cohort_retention,
    )

    return (
        customer_ltv,
        cohort_retention,
        cohort_summary,
        customer_growth_monthly,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "Loading customer growth "
        "and value source data..."
    )

    customer_master = pd.read_parquet(
        CUSTOMER_MASTER_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    customer_value = pd.read_parquet(
        CUSTOMER_VALUE_FILE
    )

    print(
        "Building customer growth, "
        "cohort and LTV analytics..."
    )

    (
        customer_ltv,
        cohort_retention,
        cohort_summary,
        customer_growth_monthly,
    ) = build_customer_growth_value(
        customer_master,
        transactions,
        customer_value,
    )

    customer_ltv.to_parquet(
        CUSTOMER_LTV_FILE,
        index=False,
    )

    cohort_retention.to_parquet(
        COHORT_RETENTION_FILE,
        index=False,
    )

    cohort_summary.to_parquet(
        COHORT_SUMMARY_FILE,
        index=False,
    )

    customer_growth_monthly.to_parquet(
        CUSTOMER_GROWTH_MONTHLY_FILE,
        index=False,
    )

    run_qa(
        customer_ltv,
        cohort_retention,
        cohort_summary,
        customer_growth_monthly,
        transactions,
    )

    print("\nFiles created:")

    for output_file in [
        CUSTOMER_LTV_FILE,
        COHORT_RETENTION_FILE,
        COHORT_SUMMARY_FILE,
        CUSTOMER_GROWTH_MONTHLY_FILE,
    ]:
        print(output_file)


if __name__ == "__main__":
    main()
