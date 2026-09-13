from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
IDENTITY_ASSIGNMENT_SEED = 1042

N_CUSTOMERS = 20_000

START_DATE = pd.Timestamp("2023-01-01")
AS_OF_DATE = pd.Timestamp("2026-07-31")

OUTPUT_DIR = Path("data/generated")

CUSTOMER_MASTER_PATH = OUTPUT_DIR / "customer_master.parquet"
IDENTITY_RECORDS_PATH = OUTPUT_DIR / "customer_identity_records.parquet"

TRANSACTIONS_PATH = OUTPUT_DIR / "transactions.parquet"
TRANSACTION_GROUND_TRUTH_PATH = (
    OUTPUT_DIR / "transaction_ground_truth.parquet"
)

RNG = np.random.default_rng(SEED)
IDENTITY_RNG = np.random.default_rng(
    IDENTITY_ASSIGNMENT_SEED
)


# ---------------------------------------------------------------------
# Retail dimensions
# ---------------------------------------------------------------------

STATES = {
    "VIC": 0.26,
    "NSW": 0.31,
    "QLD": 0.20,
    "WA": 0.11,
    "SA": 0.07,
    "TAS": 0.02,
    "ACT": 0.02,
    "NT": 0.01,
}

CHANNELS = [
    "Store",
    "Online",
    "Click & Collect",
]

CATEGORIES = [
    "Technology",
    "Home & Living",
    "Office Supplies",
    "Furniture",
    "Print & Ink",
    "Education",
]

CUSTOMER_ARCHETYPES = {
    "Frequent Regular": 0.10,
    "Regular": 0.22,
    "Occasional": 0.28,
    "Seasonal": 0.12,
    "New": 0.10,
    "Declining": 0.08,
    "Dormant": 0.06,
    "Reactivated": 0.04,
}

SHOPPING_PERSONAS = {
    "Store Traditionalist": 0.22,
    "Digital First": 0.18,
    "Omnichannel": 0.18,
    "Promo Seeker": 0.14,
    "Big Ticket": 0.10,
    "Replenishment": 0.10,
    "Category Specialist": 0.08,
}

PERSONA_CHANNEL_WEIGHTS = {
    "Store Traditionalist": [0.88, 0.07, 0.05],
    "Digital First": [0.12, 0.76, 0.12],
    "Omnichannel": [0.38, 0.37, 0.25],
    "Promo Seeker": [0.45, 0.40, 0.15],
    "Big Ticket": [0.48, 0.42, 0.10],
    "Replenishment": [0.62, 0.25, 0.13],
    "Category Specialist": [0.55, 0.32, 0.13],
}

PERSONA_DISCOUNT_WEIGHTS = {
    "Store Traditionalist": [0.55, 0.10, 0.18, 0.08, 0.07, 0.02],
    "Digital First": [0.45, 0.10, 0.20, 0.10, 0.10, 0.05],
    "Omnichannel": [0.45, 0.10, 0.20, 0.10, 0.10, 0.05],
    "Promo Seeker": [0.12, 0.08, 0.20, 0.20, 0.25, 0.15],
    "Big Ticket": [0.48, 0.10, 0.18, 0.10, 0.10, 0.04],
    "Replenishment": [0.60, 0.10, 0.15, 0.07, 0.06, 0.02],
    "Category Specialist": [0.50, 0.10, 0.18, 0.09, 0.09, 0.04],
}

ARCHETYPE_CADENCE = {
    "Frequent Regular": (7, 3),
    "Regular": (30, 10),
    "Occasional": (75, 25),
    "Seasonal": (120, 35),
    "New": (35, 12),
    "Declining": (30, 10),
    "Dormant": (45, 15),
    "Reactivated": (40, 15),
}


# ---------------------------------------------------------------------
# Identity routing design
# ---------------------------------------------------------------------
#
# The transaction generator knows the hidden canonical customer only
# while constructing the synthetic world. That hidden identifier is
# used to select a plausible source identity record, then removed from
# the operational transaction file.
#
# The production-style transaction output contains only:
#   source_customer_id
#   source_system
#
# A separate QA-only transaction_ground_truth.parquet preserves the
# hidden mapping for synthetic validation. It must never be used by
# downstream operational analytics.

CHANNEL_SOURCE_PREFERENCE = {
    "Store": [
        "POS",
        "Loyalty",
        "Ecommerce",
    ],
    "Online": [
        "Ecommerce",
        "Loyalty",
        "POS",
    ],
    "Click & Collect": [
        "Ecommerce",
        "POS",
        "Loyalty",
    ],
}

TRANSACTION_CAPABLE_SYSTEMS = {
    "POS",
    "Ecommerce",
    "Loyalty",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def weighted_choice(
    options: dict,
    size: int,
) -> np.ndarray:
    labels = list(
        options.keys()
    )
    probabilities = list(
        options.values()
    )

    return RNG.choice(
        labels,
        size=size,
        p=probabilities,
    )


def generate_customer_master() -> pd.DataFrame:
    customer_ids = [
        f"CUST{i:06d}"
        for i in range(
            1,
            N_CUSTOMERS + 1,
        )
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "state": weighted_choice(
                STATES,
                N_CUSTOMERS,
            ),
            "archetype": weighted_choice(
                CUSTOMER_ARCHETYPES,
                N_CUSTOMERS,
            ),
            "shopping_persona": weighted_choice(
                SHOPPING_PERSONAS,
                N_CUSTOMERS,
            ),
        }
    )

    customers[
        "preferred_category"
    ] = pd.NA

    specialist_mask = customers[
        "shopping_persona"
    ].eq(
        "Category Specialist"
    )

    customers.loc[
        specialist_mask,
        "preferred_category",
    ] = RNG.choice(
        CATEGORIES,
        size=specialist_mask.sum(),
    )

    acquisition_days = RNG.integers(
        0,
        (
            AS_OF_DATE
            - START_DATE
        ).days
        + 1,
        size=N_CUSTOMERS,
    )

    customers[
        "acquisition_date"
    ] = (
        START_DATE
        + pd.to_timedelta(
            acquisition_days,
            unit="D",
        )
    )

    new_mask = customers[
        "archetype"
    ].eq(
        "New"
    )

    recent_start = (
        AS_OF_DATE
        - pd.Timedelta(
            days=180
        )
    )

    recent_days = RNG.integers(
        0,
        (
            AS_OF_DATE
            - recent_start
        ).days
        + 1,
        size=new_mask.sum(),
    )

    customers.loc[
        new_mask,
        "acquisition_date",
    ] = (
        recent_start
        + pd.to_timedelta(
            recent_days,
            unit="D",
        )
    )

    return customers


def generate_purchase_dates(
    acquisition_date: pd.Timestamp,
    archetype: str,
) -> list[pd.Timestamp]:

    mean_gap, gap_sd = (
        ARCHETYPE_CADENCE[
            archetype
        ]
    )

    purchase_dates = []

    current_date = (
        acquisition_date
        + pd.Timedelta(
            days=int(
                RNG.integers(
                    0,
                    max(
                        mean_gap,
                        2,
                    ),
                )
            )
        )
    )

    purchase_number = 0

    while (
        current_date
        <= AS_OF_DATE
    ):
        purchase_dates.append(
            current_date
        )

        purchase_number += 1

        gap = max(
            1,
            int(
                RNG.normal(
                    mean_gap,
                    gap_sd,
                )
            ),
        )

        if (
            archetype
            == "Declining"
        ):
            gap = int(
                gap
                * (
                    1
                    + min(
                        purchase_number
                        * 0.035,
                        1.5,
                    )
                )
            )

        if (
            archetype
            == "Dormant"
        ):
            dormant_cutoff = (
                AS_OF_DATE
                - pd.Timedelta(
                    days=int(
                        RNG.integers(
                            150,
                            330,
                        )
                    )
                )
            )

            if (
                current_date
                >= dormant_cutoff
            ):
                break

        if (
            archetype
            == "Seasonal"
        ):
            if (
                RNG.random()
                < 0.35
            ):
                gap += int(
                    RNG.integers(
                        30,
                        100,
                    )
                )

        if (
            archetype
            == "Reactivated"
            and purchase_number
            == 4
        ):
            gap += int(
                RNG.integers(
                    180,
                    320,
                )
            )

        current_date += (
            pd.Timedelta(
                days=gap
            )
        )

    return purchase_dates


def generate_order(
    customer_id: str,
    order_number: int,
    purchase_date: pd.Timestamp,
    state: str,
    shopping_persona: str,
    preferred_category,
) -> dict:

    channel = RNG.choice(
        CHANNELS,
        p=(
            PERSONA_CHANNEL_WEIGHTS[
                shopping_persona
            ]
        ),
    )

    if (
        shopping_persona
        == "Category Specialist"
        and pd.notna(
            preferred_category
        )
        and RNG.random()
        < 0.82
    ):
        category = (
            preferred_category
        )

    elif (
        shopping_persona
        == "Big Ticket"
    ):
        category = RNG.choice(
            CATEGORIES,
            p=[
                0.38,
                0.10,
                0.07,
                0.35,
                0.05,
                0.05,
            ],
        )

    elif (
        shopping_persona
        == "Replenishment"
    ):
        category = RNG.choice(
            CATEGORIES,
            p=[
                0.08,
                0.08,
                0.38,
                0.05,
                0.33,
                0.08,
            ],
        )

    else:
        category = RNG.choice(
            CATEGORIES
        )

    units = max(
        1,
        int(
            RNG.poisson(
                1.7
            )
        ),
    )

    category_price = {
        "Technology": 180,
        "Home & Living": 85,
        "Office Supplies": 28,
        "Furniture": 240,
        "Print & Ink": 65,
        "Education": 38,
    }

    base_price = (
        category_price[
            category
        ]
    )

    if (
        shopping_persona
        == "Big Ticket"
    ):
        base_price *= 1.65

    unit_price = max(
        5,
        RNG.lognormal(
            mean=np.log(
                base_price
            ),
            sigma=0.45,
        ),
    )

    discount_values = [
        0.00,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]

    discount_pct = RNG.choice(
        discount_values,
        p=(
            PERSONA_DISCOUNT_WEIGHTS[
                shopping_persona
            ]
        ),
    )

    gross_sales = (
        unit_price
        * units
    )

    sales = (
        gross_sales
        * (
            1
            - discount_pct
        )
    )

    margin_rate = np.clip(
        RNG.normal(
            0.32,
            0.07,
        ),
        0.12,
        0.55,
    )

    gross_margin = (
        sales
        * margin_rate
    )

    cost = (
        sales
        - gross_margin
    )

    # customer_id exists only during synthetic generation.
    # It is removed before transactions.parquet is written.
    return {
        "order_id":
            f"ORD{order_number:09d}",
        "true_customer_id":
            customer_id,
        "transaction_date":
            purchase_date,
        "state":
            state,
        "channel":
            channel,
        "category":
            category,
        "units":
            units,
        "gross_sales":
            round(
                gross_sales,
                2,
            ),
        "discount_pct":
            round(
                discount_pct,
                2,
            ),
        "net_sales":
            round(
                sales,
                2,
            ),
        "cost":
            round(
                cost,
                2,
            ),
        "gross_margin":
            round(
                gross_margin,
                2,
            ),
    }


# ---------------------------------------------------------------------
# Generate transaction economics
# ---------------------------------------------------------------------

def generate_transactions(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    transactions = []
    order_number = 1

    for customer in (
        customers.itertuples(
            index=False
        )
    ):
        purchase_dates = (
            generate_purchase_dates(
                customer.acquisition_date,
                customer.archetype,
            )
        )

        for purchase_date in (
            purchase_dates
        ):
            transactions.append(
                generate_order(
                    customer_id=(
                        customer.customer_id
                    ),
                    order_number=(
                        order_number
                    ),
                    purchase_date=(
                        purchase_date
                    ),
                    state=(
                        customer.state
                    ),
                    shopping_persona=(
                        customer.shopping_persona
                    ),
                    preferred_category=(
                        customer.preferred_category
                    ),
                )
            )

            order_number += 1

    return pd.DataFrame(
        transactions
    )


# ---------------------------------------------------------------------
# Source identity assignment
# ---------------------------------------------------------------------

def validate_identity_records(
    identity_records: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:

    required_columns = {
        "identity_record_id",
        "source_customer_id",
        "source_system",
        "transaction_eligible",
        "true_person_id",
    }

    missing = (
        required_columns
        - set(
            identity_records.columns
        )
    )

    if missing:
        raise ValueError(
            "Identity records are missing "
            f"required columns: "
            f"{sorted(missing)}"
        )

    eligible = identity_records[
        identity_records[
            "transaction_eligible"
        ].fillna(
            False
        )
    ].copy()

    invalid_systems = (
        set(
            eligible[
                "source_system"
            ].dropna().unique()
        )
        - TRANSACTION_CAPABLE_SYSTEMS
    )

    if invalid_systems:
        raise ValueError(
            "Transaction-eligible identities "
            "contain unexpected source systems: "
            f"{sorted(invalid_systems)}"
        )

    customer_ids = set(
        customers[
            "customer_id"
        ].astype(
            str
        )
    )

    eligible_people = set(
        eligible[
            "true_person_id"
        ].astype(
            str
        )
    )

    uncovered = (
        customer_ids
        - eligible_people
    )

    if uncovered:
        raise ValueError(
            f"{len(uncovered):,} customers "
            "have no transaction-capable "
            "source identity. Regenerate "
            "identity records using the "
            "transaction-aware identity "
            "generator before generating "
            "transactions."
        )

    duplicate_source_keys = (
        identity_records[
            "source_customer_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_source_keys:
        raise ValueError(
            "source_customer_id must be "
            "globally unique. Found "
            f"{duplicate_source_keys:,} "
            "duplicates."
        )


def build_identity_lookup(
    identity_records: pd.DataFrame,
) -> dict[str, dict[str, list[str]]]:

    eligible = identity_records[
        identity_records[
            "transaction_eligible"
        ].fillna(
            False
        )
    ].copy()

    eligible[
        "true_person_id"
    ] = eligible[
        "true_person_id"
    ].astype(
        str
    )

    lookup = {}

    for true_person_id, group in (
        eligible.groupby(
            "true_person_id",
            sort=False,
        )
    ):
        by_system = {}

        for source_system, system_group in (
            group.groupby(
                "source_system",
                sort=False,
            )
        ):
            by_system[
                source_system
            ] = (
                system_group[
                    "source_customer_id"
                ]
                .astype(
                    str
                )
                .tolist()
            )

        lookup[
            true_person_id
        ] = by_system

    return lookup


def choose_source_identity(
    true_customer_id: str,
    channel: str,
    identity_lookup: dict,
) -> tuple[str, str]:

    by_system = (
        identity_lookup[
            str(
                true_customer_id
            )
        ]
    )

    preference = (
        CHANNEL_SOURCE_PREFERENCE[
            channel
        ]
    )

    for source_system in (
        preference
    ):
        candidates = (
            by_system.get(
                source_system,
                [],
            )
        )

        if candidates:
            selected = (
                IDENTITY_RNG.choice(
                    candidates
                )
            )

            return (
                str(
                    selected
                ),
                source_system,
            )

    # Defensive fallback. This should never be reached because
    # every customer is guaranteed at least one transaction-
    # capable identity and the preference list covers all three
    # eligible systems.
    all_candidates = [
        (
            source_system,
            source_customer_id,
        )
        for (
            source_system,
            source_ids,
        ) in by_system.items()
        for source_customer_id in (
            source_ids
        )
    ]

    if not all_candidates:
        raise ValueError(
            "No transaction-capable identity "
            f"available for "
            f"{true_customer_id}."
        )

    selected_index = int(
        IDENTITY_RNG.integers(
            0,
            len(
                all_candidates
            ),
        )
    )

    source_system, source_customer_id = (
        all_candidates[
            selected_index
        ]
    )

    return (
        str(
            source_customer_id
        ),
        str(
            source_system
        ),
    )


def assign_source_identities(
    transactions_with_truth: pd.DataFrame,
    identity_records: pd.DataFrame,
) -> pd.DataFrame:

    identity_lookup = (
        build_identity_lookup(
            identity_records
        )
    )

    assigned_source_customer_ids = []
    assigned_source_systems = []

    for row in (
        transactions_with_truth[
            [
                "true_customer_id",
                "channel",
            ]
        ].itertuples(
            index=False
        )
    ):
        (
            source_customer_id,
            source_system,
        ) = choose_source_identity(
            true_customer_id=(
                row.true_customer_id
            ),
            channel=row.channel,
            identity_lookup=(
                identity_lookup
            ),
        )

        assigned_source_customer_ids.append(
            source_customer_id
        )

        assigned_source_systems.append(
            source_system
        )

    output = (
        transactions_with_truth.copy()
    )

    output[
        "source_customer_id"
    ] = (
        assigned_source_customer_ids
    )

    output[
        "source_system"
    ] = (
        assigned_source_systems
    )

    return output


# ---------------------------------------------------------------------
# Operational / QA output split
# ---------------------------------------------------------------------

def split_outputs(
    transactions_with_identity: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    ground_truth = (
        transactions_with_identity[
            [
                "order_id",
                "true_customer_id",
                "source_customer_id",
                "source_system",
            ]
        ]
        .copy()
    )

    operational = (
        transactions_with_identity.drop(
            columns=[
                "true_customer_id",
            ]
        )
        .copy()
    )

    ordered_columns = [
        "order_id",
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

    operational = (
        operational[
            ordered_columns
        ]
    )

    return (
        operational,
        ground_truth,
    )


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    transaction_ground_truth: pd.DataFrame,
    identity_records: pd.DataFrame,
) -> None:

    print()
    print("=" * 68)
    print(
        "TRANSACTION GENERATION COMPLETE"
    )
    print("=" * 68)

    print()
    print("CUSTOMER QA")
    print("-" * 68)

    print(
        f"Hidden synthetic customers     : "
        f"{len(customers):,}"
    )

    print(
        f"Acquisition range              : "
        f"{customers['acquisition_date'].min().date()} "
        f"to "
        f"{customers['acquisition_date'].max().date()}"
    )

    print()
    print(
        "Underlying behavioural mix"
    )
    print("-" * 68)

    print(
        customers[
            "archetype"
        ]
        .value_counts(
            normalize=True
        )
        .mul(
            100
        )
        .round(
            1
        )
        .astype(
            str
        )
        + "%"
    )

    print()
    print(
        "Underlying shopping persona mix"
    )
    print("-" * 68)

    print(
        customers[
            "shopping_persona"
        ]
        .value_counts(
            normalize=True
        )
        .mul(
            100
        )
        .round(
            1
        )
        .astype(
            str
        )
        + "%"
    )

    print()
    print("TRANSACTION QA")
    print("-" * 68)

    print(
        f"Orders                         : "
        f"{len(transactions):,}"
    )

    print(
        f"Transaction range              : "
        f"{transactions['transaction_date'].min().date()} "
        f"to "
        f"{transactions['transaction_date'].max().date()}"
    )

    print(
        f"Source identities used         : "
        f"{transactions['source_customer_id'].nunique():,}"
    )

    print(
        f"Source systems used            : "
        f"{transactions['source_system'].nunique():,}"
    )

    print(
        f"Net sales                      : "
        f"${transactions['net_sales'].sum():,.0f}"
    )

    print(
        f"Gross margin                   : "
        f"${transactions['gross_margin'].sum():,.0f}"
    )

    print(
        f"Average order value            : "
        f"${transactions['net_sales'].mean():,.2f}"
    )

    print()
    print("IDENTITY BRIDGE QA")
    print("-" * 68)

    mapped_orders = (
        transactions[
            "source_customer_id"
        ]
        .notna()
        .sum()
    )

    print(
        f"Orders mapped to source ID     : "
        f"{mapped_orders:,}"
    )

    print(
        f"Source identity coverage       : "
        f"{mapped_orders / len(transactions):.1%}"
    )

    print(
        f"Hidden customer field in ops   : "
        f"{'true_customer_id' in transactions.columns}"
    )

    print(
        f"Ground-truth rows              : "
        f"{len(transaction_ground_truth):,}"
    )

    truth_customer_count = (
        transaction_ground_truth[
            "true_customer_id"
        ]
        .nunique()
    )

    print(
        f"Hidden purchasing customers    : "
        f"{truth_customer_count:,}"
    )

    valid_source_ids = set(
        identity_records[
            "source_customer_id"
        ].astype(
            str
        )
    )

    mapped_source_ids = set(
        transactions[
            "source_customer_id"
        ].astype(
            str
        )
    )

    orphan_source_ids = (
        mapped_source_ids
        - valid_source_ids
    )

    print(
        f"Orphan transaction source IDs : "
        f"{len(orphan_source_ids):,}"
    )

    print()
    print(
        "Source identity used by channel"
    )
    print("-" * 68)

    source_mix = (
        transactions.groupby(
            [
                "channel",
                "source_system",
            ]
        )
        .size()
        .rename(
            "orders"
        )
        .reset_index()
    )

    source_mix[
        "share_within_channel"
    ] = (
        source_mix[
            "orders"
        ]
        / source_mix.groupby(
            "channel"
        )[
            "orders"
        ].transform(
            "sum"
        )
    )

    source_mix[
        "share_within_channel"
    ] = (
        source_mix[
            "share_within_channel"
        ]
        .map(
            lambda value:
                f"{value:.1%}"
        )
    )

    print(
        source_mix.to_string(
            index=False
        )
    )

    print()
    print(
        f"Operational transactions: "
        f"{TRANSACTIONS_PATH}"
    )

    print(
        f"Transaction ground truth: "
        f"{TRANSACTION_GROUND_TRUTH_PATH}"
    )

    print("=" * 68)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Recreate the deterministic customer master so the retail
    # transaction economics remain reproducible.
    print(
        "Generating customer master..."
    )

    customers = (
        generate_customer_master()
    )

    customers.to_parquet(
        CUSTOMER_MASTER_PATH,
        index=False,
    )

    if not (
        IDENTITY_RECORDS_PATH.exists()
    ):
        raise FileNotFoundError(
            "Transaction-aware identity records "
            "are required before transaction "
            "generation. Run:\n"
            "python -m "
            "src.data_generation."
            "generate_identity_records"
        )

    print(
        "Loading transaction-aware "
        "identity records..."
    )

    identity_records = (
        pd.read_parquet(
            IDENTITY_RECORDS_PATH
        )
    )

    validate_identity_records(
        identity_records=(
            identity_records
        ),
        customers=customers,
    )

    print(
        "Generating transaction economics..."
    )

    transactions_with_truth = (
        generate_transactions(
            customers
        )
    )

    print(
        "Assigning operational "
        "source identities..."
    )

    transactions_with_identity = (
        assign_source_identities(
            transactions_with_truth=(
                transactions_with_truth
            ),
            identity_records=(
                identity_records
            ),
        )
    )

    (
        transactions,
        transaction_ground_truth,
    ) = split_outputs(
        transactions_with_identity
    )

    transactions.to_parquet(
        TRANSACTIONS_PATH,
        index=False,
    )

    transaction_ground_truth.to_parquet(
        TRANSACTION_GROUND_TRUTH_PATH,
        index=False,
    )

    run_qa(
        customers=customers,
        transactions=transactions,
        transaction_ground_truth=(
            transaction_ground_truth
        ),
        identity_records=(
            identity_records
        ),
    )


if __name__ == "__main__":
    main()
