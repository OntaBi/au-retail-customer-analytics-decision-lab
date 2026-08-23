from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
N_CUSTOMERS = 20_000

START_DATE = pd.Timestamp("2023-01-01")
AS_OF_DATE = pd.Timestamp("2026-07-31")

OUTPUT_DIR = Path("data/generated")

RNG = np.random.default_rng(SEED)


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

CHANNELS = ["Store", "Online", "Click & Collect"]

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
# Helpers
# ---------------------------------------------------------------------

def weighted_choice(options: dict, size: int) -> np.ndarray:
    labels = list(options.keys())
    probabilities = list(options.values())

    return RNG.choice(
        labels,
        size=size,
        p=probabilities,
    )


def generate_customer_master() -> pd.DataFrame:
    customer_ids = [
        f"CUST{i:06d}"
        for i in range(1, N_CUSTOMERS + 1)
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "state": weighted_choice(STATES, N_CUSTOMERS),
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

    # Category Specialists have a persistent preferred category.
    customers["preferred_category"] = pd.NA

    specialist_mask = customers[
        "shopping_persona"
    ].eq("Category Specialist")

    customers.loc[
        specialist_mask,
        "preferred_category",
    ] = RNG.choice(
        CATEGORIES,
        size=specialist_mask.sum(),
    )

    acquisition_days = RNG.integers(
        0,
        (AS_OF_DATE - START_DATE).days + 1,
        size=N_CUSTOMERS,
    )

    customers["acquisition_date"] = (
        START_DATE
        + pd.to_timedelta(acquisition_days, unit="D")
    )

    # New customers should genuinely be recent acquisitions.
    new_mask = customers["archetype"].eq("New")

    recent_start = AS_OF_DATE - pd.Timedelta(days=180)

    recent_days = RNG.integers(
        0,
        (AS_OF_DATE - recent_start).days + 1,
        size=new_mask.sum(),
    )

    customers.loc[new_mask, "acquisition_date"] = (
        recent_start
        + pd.to_timedelta(recent_days, unit="D")
    )

    return customers


def generate_purchase_dates(
    acquisition_date: pd.Timestamp,
    archetype: str,
) -> list[pd.Timestamp]:

    mean_gap, gap_sd = ARCHETYPE_CADENCE[archetype]

    purchase_dates = []

    current_date = acquisition_date + pd.Timedelta(
        days=int(RNG.integers(0, max(mean_gap, 2)))
    )

    purchase_number = 0

    while current_date <= AS_OF_DATE:

        purchase_dates.append(current_date)
        purchase_number += 1

        gap = max(
            1,
            int(RNG.normal(mean_gap, gap_sd)),
        )

        # Declining customers progressively shop less frequently.
        if archetype == "Declining":
            gap = int(
                gap
                * (
                    1
                    + min(
                        purchase_number * 0.035,
                        1.5,
                    )
                )
            )

        # Dormant customers stop shopping well before the as-of date.
        if archetype == "Dormant":
            dormant_cutoff = (
                AS_OF_DATE
                - pd.Timedelta(
                    days=int(RNG.integers(150, 330))
                )
            )

            if current_date >= dormant_cutoff:
                break

        # Seasonal customers have more irregular purchase intervals.
        if archetype == "Seasonal":
            if RNG.random() < 0.35:
                gap += int(RNG.integers(30, 100))

        # Reactivated customers experience a long absence followed
        # by a return to purchasing.
        if (
            archetype == "Reactivated"
            and purchase_number == 4
        ):
            gap += int(RNG.integers(180, 320))

        current_date += pd.Timedelta(days=gap)

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
        p=PERSONA_CHANNEL_WEIGHTS[
            shopping_persona
        ],
    )

    if (
        shopping_persona == "Category Specialist"
        and pd.notna(preferred_category)
        and RNG.random() < 0.82
    ):
        category = preferred_category

    elif shopping_persona == "Big Ticket":
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

    elif shopping_persona == "Replenishment":
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
        int(RNG.poisson(1.7)),
    )

    category_price = {
        "Technology": 180,
        "Home & Living": 85,
        "Office Supplies": 28,
        "Furniture": 240,
        "Print & Ink": 65,
        "Education": 38,
    }

    base_price = category_price[
        category
    ]

    if shopping_persona == "Big Ticket":
        base_price *= 1.65

    unit_price = max(
        5,
        RNG.lognormal(
            mean=np.log(base_price),
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
        p=PERSONA_DISCOUNT_WEIGHTS[
            shopping_persona
        ],
    )

    gross_sales = (
        unit_price * units
    )

    sales = (
        gross_sales
        * (1 - discount_pct)
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
        sales * margin_rate
    )

    cost = (
        sales - gross_margin
    )

    return {
        "order_id":
            f"ORD{order_number:09d}",
        "customer_id":
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
            round(gross_sales, 2),
        "discount_pct":
            round(discount_pct, 2),
        "net_sales":
            round(sales, 2),
        "cost":
            round(cost, 2),
        "gross_margin":
            round(gross_margin, 2),
    }


# ---------------------------------------------------------------------
# Generate transactions
# ---------------------------------------------------------------------

def generate_transactions(
    customers: pd.DataFrame,
) -> pd.DataFrame:

    transactions = []
    order_number = 1

    for customer in customers.itertuples(index=False):

        purchase_dates = generate_purchase_dates(
            customer.acquisition_date,
            customer.archetype,
        )

        for purchase_date in purchase_dates:

            transactions.append(
                generate_order(
                    customer_id=customer.customer_id,
                    order_number=order_number,
                    purchase_date=purchase_date,
                    state=customer.state,
                    shopping_persona=customer.shopping_persona,
                    preferred_category=customer.preferred_category,
                )
            )

            order_number += 1

    return pd.DataFrame(transactions)


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:

    print("\nCUSTOMER QA")
    print("-" * 60)

    print(f"Customers: {len(customers):,}")
    print(
        f"Acquisition range: "
        f"{customers['acquisition_date'].min().date()} "
        f"to "
        f"{customers['acquisition_date'].max().date()}"
    )

    print("\nUnderlying behavioural mix:")
    print(
        customers["archetype"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    print("\nUnderlying shopping persona mix:")
    print(
        customers["shopping_persona"]
        .value_counts(normalize=True)
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    print("\nTRANSACTION QA")
    print("-" * 60)

    print(f"Orders: {len(transactions):,}")

    print(
        f"Transaction range: "
        f"{transactions['transaction_date'].min().date()} "
        f"to "
        f"{transactions['transaction_date'].max().date()}"
    )

    print(
        f"Customers with transactions: "
        f"{transactions['customer_id'].nunique():,}"
    )

    print(
        f"Net sales: "
        f"${transactions['net_sales'].sum():,.0f}"
    )

    print(
        f"Gross margin: "
        f"${transactions['gross_margin'].sum():,.0f}"
    )

    print(
        f"Average order value: "
        f"${transactions['net_sales'].mean():,.2f}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating customer master...")

    customers = generate_customer_master()

    print("Generating transactions...")

    transactions = generate_transactions(customers)

    customers.to_parquet(
        OUTPUT_DIR / "customer_master.parquet",
        index=False,
    )

    transactions.to_parquet(
        OUTPUT_DIR / "transactions.parquet",
        index=False,
    )

    run_qa(
        customers,
        transactions,
    )

    print("\nFiles created:")
    print(
        OUTPUT_DIR / "customer_master.parquet"
    )
    print(
        OUTPUT_DIR / "transactions.parquet"
    )


if __name__ == "__main__":
    main()