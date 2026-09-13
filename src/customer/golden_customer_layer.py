from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

TRANSACTIONS_PATH = (
    ROOT / "data" / "generated" / "transactions.parquet"
)

IDENTITY_RESOLUTION_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "customer_identity_resolution.parquet"
)

IDENTITY_RECORDS_PATH = (
    ROOT
    / "data"
    / "generated"
    / "customer_identity_records.parquet"
)

OUTPUT_TRANSACTION_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "golden_customer_transactions.parquet"
)

OUTPUT_MASTER_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "golden_customer_master.parquet"
)

OUTPUT_RECONCILIATION_PATH = (
    ROOT
    / "data"
    / "runtime"
    / "golden_customer_reconciliation.parquet"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: "
            f"{sorted(missing)}"
        )


def load_inputs():
    if not TRANSACTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Transactions not found: "
            f"{TRANSACTIONS_PATH}"
        )

    if not IDENTITY_RESOLUTION_PATH.exists():
        raise FileNotFoundError(
            f"Identity resolution output not found: "
            f"{IDENTITY_RESOLUTION_PATH}"
        )

    if not IDENTITY_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Identity source records not found: "
            f"{IDENTITY_RECORDS_PATH}"
        )

    transactions = pd.read_parquet(
        TRANSACTIONS_PATH
    )

    resolution = pd.read_parquet(
        IDENTITY_RESOLUTION_PATH
    )

    identity_records = pd.read_parquet(
        IDENTITY_RECORDS_PATH
    )

    return (
        transactions,
        resolution,
        identity_records,
    )


def prepare_resolution_lookup(
    resolution: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        resolution,
        {
            "identity_record_id",
            "golden_customer_id",
        },
        "Identity resolution output",
    )

    lookup = (
        resolution[
            [
                "identity_record_id",
                "golden_customer_id",
            ]
        ]
        .copy()
        .rename(
            columns={
                "identity_record_id":
                    "source_customer_id"
            }
        )
    )

    if lookup[
        "source_customer_id"
    ].duplicated().any():
        duplicate_count = (
            lookup[
                "source_customer_id"
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Identity resolution lookup must "
            "contain one row per source identity. "
            f"Found {duplicate_count:,} duplicate "
            "source_customer_id values."
        )

    return lookup


def build_golden_transactions(
    transactions: pd.DataFrame,
    resolution_lookup: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        transactions,
        {
            "order_id",
            "source_customer_id",
            "source_system",
            "transaction_date",
            "net_sales",
            "gross_margin",
        },
        "Operational transactions",
    )

    merged = transactions.merge(
        resolution_lookup,
        on="source_customer_id",
        how="left",
        validate="many_to_one",
    )

    unresolved_orders = (
        merged[
            "golden_customer_id"
        ]
        .isna()
        .sum()
    )

    if unresolved_orders:
        raise ValueError(
            f"{unresolved_orders:,} transaction "
            "rows could not be mapped to a resolved "
            "golden customer."
        )

    ordered_columns = [
        "order_id",
        "golden_customer_id",
        "source_customer_id",
        "source_system",
        "transaction_date",
        "state",
        "channel",
        "category",
        "units",
        "gross_sales",
        "discount_pct",
        "net_sales",
        "cost",
        "gross_margin",
    ]

    existing_columns = [
        column
        for column in ordered_columns
        if column in merged.columns
    ]

    return (
        merged[
            existing_columns
        ]
        .copy()
    )


def build_golden_master(
    identity_records: pd.DataFrame,
    resolution_lookup: pd.DataFrame,
) -> pd.DataFrame:

    require_columns(
        identity_records,
        {
            "identity_record_id",
            "source_system",
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "postcode",
            "state",
        },
        "Identity source records",
    )

    # Production-style projection:
    # deliberately exclude hidden truth / synthetic-only fields.
    observable_columns = [
        "identity_record_id",
        "source_system",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "postcode",
        "state",
        "source_role",
        "transaction_eligible",
    ]

    observable_columns = [
        column
        for column in observable_columns
        if column in identity_records.columns
    ]

    source = (
        identity_records[
            observable_columns
        ]
        .copy()
        .rename(
            columns={
                "identity_record_id":
                    "source_customer_id"
            }
        )
    )

    resolved = source.merge(
        resolution_lookup,
        on="source_customer_id",
        how="inner",
        validate="one_to_one",
    )

    if resolved.empty:
        raise ValueError(
            "No resolved identity records were "
            "available to build the golden customer "
            "master."
        )

    # Choose a representative source record for human-readable
    # profile fields. Stronger / more operational roles take priority.
    role_priority = {
        "Primary Transaction Identity": 1,
        "Additional Transaction Identity": 2,
        "Supporting Identity": 3,
    }

    resolved[
        "_role_priority"
    ] = (
        resolved[
            "source_role"
        ]
        .map(
            role_priority
        )
        .fillna(
            9
        )
        if "source_role" in resolved.columns
        else 9
    )

    completeness_fields = [
        field
        for field in [
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "postcode",
            "state",
        ]
        if field in resolved.columns
    ]

    resolved[
        "_completeness"
    ] = (
        resolved[
            completeness_fields
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    representative = (
        resolved
        .sort_values(
            [
                "golden_customer_id",
                "_role_priority",
                "_completeness",
                "source_customer_id",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .drop_duplicates(
            "golden_customer_id"
        )
        .copy()
    )

    aggregation = (
        resolved.groupby(
            "golden_customer_id",
            as_index=False,
        )
        .agg(
            source_records_linked=(
                "source_customer_id",
                "nunique",
            ),
            source_systems=(
                "source_system",
                lambda values:
                    " | ".join(
                        sorted(
                            set(
                                values.dropna().astype(str)
                            )
                        )
                    ),
            ),
        )
    )

    master_columns = [
        "golden_customer_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "postcode",
        "state",
        "source_customer_id",
        "source_system",
    ]

    master_columns = [
        column
        for column in master_columns
        if column in representative.columns
    ]

    representative = (
        representative[
            master_columns
        ]
        .rename(
            columns={
                "source_customer_id":
                    "representative_source_customer_id",
                "source_system":
                    "representative_source_system",
            }
        )
    )

    golden_master = (
        representative.merge(
            aggregation,
            on="golden_customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    return golden_master


def build_reconciliation(
    transactions: pd.DataFrame,
    golden_transactions: pd.DataFrame,
    golden_master: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    def add_metric(
        metric,
        before,
        after,
    ):
        rows.append(
            {
                "metric": metric,
                "before_value": before,
                "after_value": after,
                "difference": after - before,
            }
        )

    add_metric(
        "Orders",
        len(transactions),
        len(golden_transactions),
    )

    add_metric(
        "Net Sales",
        float(
            transactions[
                "net_sales"
            ].sum()
        ),
        float(
            golden_transactions[
                "net_sales"
            ].sum()
        ),
    )

    add_metric(
        "Gross Margin",
        float(
            transactions[
                "gross_margin"
            ].sum()
        ),
        float(
            golden_transactions[
                "gross_margin"
            ].sum()
        ),
    )

    add_metric(
        "Source Identities Used",
        transactions[
            "source_customer_id"
        ].nunique(),
        golden_transactions[
            "source_customer_id"
        ].nunique(),
    )

    add_metric(
        "Golden Customers With Purchases",
        0,
        golden_transactions[
            "golden_customer_id"
        ].nunique(),
    )

    add_metric(
        "Resolved Golden Customer Population",
        0,
        golden_master[
            "golden_customer_id"
        ].nunique(),
    )

    return pd.DataFrame(
        rows
    )


def print_summary(
    transactions: pd.DataFrame,
    golden_transactions: pd.DataFrame,
    golden_master: pd.DataFrame,
    reconciliation: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print("GOLDEN CUSTOMER LAYER COMPLETE")
    print("=" * 72)

    print(
        f"Operational transaction rows      : "
        f"{len(transactions):,}"
    )

    print(
        f"Golden transaction rows           : "
        f"{len(golden_transactions):,}"
    )

    print(
        f"Resolved golden customer population: "
        f"{golden_master['golden_customer_id'].nunique():,}"
    )

    print(
        f"Golden customers with purchases   : "
        f"{golden_transactions['golden_customer_id'].nunique():,}"
    )

    print(
        f"Source identities used in orders  : "
        f"{golden_transactions['source_customer_id'].nunique():,}"
    )

    print()
    print("COMMERCIAL RECONCILIATION")
    print("-" * 72)

    commercial_metrics = (
        reconciliation[
            reconciliation[
                "metric"
            ].isin(
                [
                    "Orders",
                    "Net Sales",
                    "Gross Margin",
                ]
            )
        ]
    )

    for row in (
        commercial_metrics.itertuples(
            index=False
        )
    ):
        if row.metric in {
            "Net Sales",
            "Gross Margin",
        }:
            print(
                f"{row.metric:<28}: "
                f"${row.before_value:,.2f} "
                f"→ "
                f"${row.after_value:,.2f} "
                f"(diff ${row.difference:,.2f})"
            )
        else:
            print(
                f"{row.metric:<28}: "
                f"{int(row.before_value):,} "
                f"→ "
                f"{int(row.after_value):,} "
                f"(diff {int(row.difference):,})"
            )

    print()
    print("OUTPUTS")
    print("-" * 72)

    print(
        f"Golden transactions : "
        f"{OUTPUT_TRANSACTION_PATH}"
    )

    print(
        f"Golden master       : "
        f"{OUTPUT_MASTER_PATH}"
    )

    print(
        f"Reconciliation      : "
        f"{OUTPUT_RECONCILIATION_PATH}"
    )

    print("=" * 72)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    (
        transactions,
        resolution,
        identity_records,
    ) = load_inputs()

    resolution_lookup = (
        prepare_resolution_lookup(
            resolution
        )
    )

    golden_transactions = (
        build_golden_transactions(
            transactions=transactions,
            resolution_lookup=resolution_lookup,
        )
    )

    golden_master = (
        build_golden_master(
            identity_records=identity_records,
            resolution_lookup=resolution_lookup,
        )
    )

    reconciliation = (
        build_reconciliation(
            transactions=transactions,
            golden_transactions=golden_transactions,
            golden_master=golden_master,
        )
    )

    OUTPUT_TRANSACTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    golden_transactions.to_parquet(
        OUTPUT_TRANSACTION_PATH,
        index=False,
    )

    golden_master.to_parquet(
        OUTPUT_MASTER_PATH,
        index=False,
    )

    reconciliation.to_parquet(
        OUTPUT_RECONCILIATION_PATH,
        index=False,
    )

    print_summary(
        transactions=transactions,
        golden_transactions=golden_transactions,
        golden_master=golden_master,
        reconciliation=reconciliation,
    )


if __name__ == "__main__":
    main()
