from pathlib import Path

from src.data_generation.generate_transactions import (
    CUSTOMER_MASTER_PATH,
    generate_customer_master,
)


def main() -> None:
    """
    Generate the deterministic synthetic customer master before
    identity and transaction generation.

    This breaks the data-generation dependency cycle cleanly:

        customer master
            -> identity source records
            -> operational transactions

    generate_transactions.py intentionally regenerates the same
    deterministic master internally before creating transaction
    economics, so its RNG sequence and retail economics remain
    reproducible.
    """

    CUSTOMER_MASTER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating base synthetic customer master...")

    customers = generate_customer_master()

    customers.to_parquet(
        CUSTOMER_MASTER_PATH,
        index=False,
    )

    print()
    print("=" * 68)
    print("CUSTOMER MASTER GENERATION COMPLETE")
    print("=" * 68)

    print(
        f"Customers          : "
        f"{len(customers):,}"
    )

    print(
        f"Unique customer IDs: "
        f"{customers['customer_id'].nunique():,}"
    )

    print(
        f"Acquisition range  : "
        f"{customers['acquisition_date'].min().date()} "
        f"to "
        f"{customers['acquisition_date'].max().date()}"
    )

    print()
    print(
        f"Output: "
        f"{CUSTOMER_MASTER_PATH}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()
