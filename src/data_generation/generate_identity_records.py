from pathlib import Path
import random
import re
import string

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[2]

CUSTOMER_MASTER_PATH = ROOT / "data" / "generated" / "customer_master.parquet"

OUTPUT_IDENTITY_PATH = (
    ROOT / "data" / "generated" / "customer_identity_records.parquet"
)

OUTPUT_GROUND_TRUTH_PATH = (
    ROOT / "data" / "generated" / "identity_ground_truth.parquet"
)


# ============================================================
# Helpers
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# Source-system design
# ============================================================
#
# Every customer receives one operational transaction-capable
# identity. Additional noisy identities may appear across other
# enterprise systems. This lets downstream transaction activity
# be mapped to source identities without using hidden ground truth.
#
# POS and Ecommerce are direct transaction systems. Loyalty is
# treated as transaction-capable because a recognised loyalty /
# account identifier can also be captured against an order.
# Service and Marketing contribute identity evidence but do not
# independently own retail transactions.

TRANSACTION_CAPABLE_SYSTEMS = [
    "POS",
    "Ecommerce",
    "Loyalty",
]

NON_TRANSACTION_SYSTEMS = [
    "Service",
    "Marketing",
]

ALL_SOURCE_SYSTEMS = (
    TRANSACTION_CAPABLE_SYSTEMS
    + NON_TRANSACTION_SYSTEMS
)

PRIMARY_SOURCE_WEIGHTS = [
    0.45,  # POS
    0.35,  # Ecommerce
    0.20,  # Loyalty
]


def normalise_string(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def random_typo(value):
    """
    Introduce a light synthetic typo:
    - delete one character
    - swap adjacent characters
    - replace one character
    """

    value = normalise_string(value)

    if not value or len(value) < 4:
        return value

    choice = random.choice(["delete", "swap", "replace"])

    idx = random.randint(1, len(value) - 2)

    if choice == "delete":
        return value[:idx] + value[idx + 1 :]

    if choice == "swap":
        chars = list(value)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    replacement = random.choice(string.ascii_lowercase)
    return value[:idx] + replacement + value[idx + 1 :]


def alter_email(email):
    email = normalise_string(email)

    if not email or "@" not in email:
        return email

    local, domain = email.split("@", 1)

    choice = random.choice(
        [
            "lower",
            "upper",
            "space",
            "dot_local",
            "plus_alias",
            "typo",
        ]
    )

    if choice == "lower":
        return email.lower()

    if choice == "upper":
        return email.upper()

    if choice == "space":
        return f" {email} "

    if choice == "dot_local" and len(local) > 4:
        idx = random.randint(1, len(local) - 2)
        local = local[:idx] + "." + local[idx:]
        return f"{local}@{domain}"

    if choice == "plus_alias":
        return f"{local}+shop@{domain}"

    if choice == "typo":
        return random_typo(email)

    return email


def digits_only(value):
    if pd.isna(value):
        return None

    digits = re.sub(r"\D", "", str(value))

    if not digits:
        return None

    return digits


def alter_phone(phone):
    phone = normalise_string(phone)

    if not phone:
        return phone

    digits = digits_only(phone)

    if not digits:
        return phone

    choice = random.choice(
        [
            "spaces",
            "compact",
            "country_code",
            "drop_digit",
        ]
    )

    if choice == "spaces":
        if len(digits) >= 10:
            return f"{digits[:4]} {digits[4:7]} {digits[7:]}"
        return phone

    if choice == "compact":
        return digits

    if choice == "country_code":
        if digits.startswith("0") and len(digits) >= 9:
            return "+61" + digits[1:]
        return phone

    if choice == "drop_digit" and len(digits) > 7:
        idx = random.randint(2, len(digits) - 2)
        return digits[:idx] + digits[idx + 1 :]

    return phone


def alter_address(address):
    address = normalise_string(address)

    if not address:
        return address

    replacements = {
        "Street": "St",
        "Road": "Rd",
        "Avenue": "Ave",
        "Drive": "Dr",
        "Court": "Ct",
        "Boulevard": "Blvd",
        "Lane": "Ln",
        "Place": "Pl",
    }

    choice = random.choice(
        [
            "abbreviate",
            "lower",
            "upper",
            "remove_space",
            "typo",
        ]
    )

    if choice == "abbreviate":
        updated = address

        for full, short in replacements.items():
            updated = re.sub(
                rf"\b{full}\b",
                short,
                updated,
                flags=re.IGNORECASE,
            )

        return updated

    if choice == "lower":
        return address.lower()

    if choice == "upper":
        return address.upper()

    if choice == "remove_space":
        return re.sub(r"\s+", " ", address).strip()

    if choice == "typo":
        return random_typo(address)

    return address


def alter_name(name):
    name = normalise_string(name)

    if not name:
        return name

    choice = random.choice(
        [
            "lower",
            "upper",
            "typo",
            "initial",
        ]
    )

    if choice == "lower":
        return name.lower()

    if choice == "upper":
        return name.upper()

    if choice == "typo":
        return random_typo(name)

    if choice == "initial":
        return name[0] if len(name) > 0 else name

    return name


# ============================================================
# Source-field discovery
# ============================================================

def find_column(df, candidates):
    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def detect_source_columns(df):
    """
    Find likely customer master columns without assuming
    an exact schema.
    """

    columns = {}

    columns["customer_id"] = find_column(
        df,
        [
            "customer_id",
            "customerid",
            "cust_id",
            "customer_key",
        ],
    )

    columns["first_name"] = find_column(
        df,
        [
            "first_name",
            "firstname",
            "given_name",
        ],
    )

    columns["last_name"] = find_column(
        df,
        [
            "last_name",
            "lastname",
            "surname",
            "family_name",
        ],
    )

    columns["email"] = find_column(
        df,
        [
            "email",
            "email_address",
        ],
    )

    columns["phone"] = find_column(
        df,
        [
            "phone",
            "phone_number",
            "mobile",
            "mobile_number",
        ],
    )

    columns["address"] = find_column(
        df,
        [
            "address",
            "street_address",
            "address_line_1",
        ],
    )

    columns["postcode"] = find_column(
        df,
        [
            "postcode",
            "post_code",
            "postal_code",
        ],
    )

    columns["state"] = find_column(
        df,
        [
            "state",
            "customer_state",
        ],
    )

    return columns


# ============================================================
# Synthetic fallback attributes
# ============================================================

FIRST_NAMES = [
    "Olivia",
    "Charlotte",
    "Amelia",
    "Isla",
    "Mia",
    "Ava",
    "Grace",
    "Sophie",
    "Jack",
    "Noah",
    "Oliver",
    "William",
    "Henry",
    "Leo",
    "Ethan",
    "Lucas",
    "Aarav",
    "Arjun",
    "Priya",
    "Anika",
    "Wei",
    "Mei",
    "Daniel",
    "Sarah",
    "Michael",
    "Emma",
]

LAST_NAMES = [
    "Smith",
    "Jones",
    "Williams",
    "Brown",
    "Wilson",
    "Taylor",
    "Johnson",
    "Lee",
    "Chen",
    "Singh",
    "Patel",
    "Nguyen",
    "Martin",
    "White",
    "Anderson",
    "Thomas",
    "Thompson",
    "Walker",
    "Harris",
    "King",
]


def build_fallback_identity(customer_id, state=None):
    seed = abs(hash(str(customer_id))) % (2**32)

    rng = np.random.default_rng(seed)

    first_name = rng.choice(FIRST_NAMES)
    last_name = rng.choice(LAST_NAMES)

    domain = rng.choice(
        [
            "gmail.com",
            "outlook.com",
            "hotmail.com",
            "icloud.com",
            "yahoo.com.au",
        ]
    )

    email = (
        f"{first_name.lower()}."
        f"{last_name.lower()}"
        f"{rng.integers(10, 999)}"
        f"@{domain}"
    )

    mobile_suffix = str(rng.integers(10_000_000, 99_999_999))

    phone = "04" + mobile_suffix

    street_number = rng.integers(1, 250)

    street_name = rng.choice(
        [
            "King",
            "High",
            "Station",
            "Victoria",
            "George",
            "Park",
            "Church",
            "Albert",
            "Spring",
            "Main",
        ]
    )

    street_type = rng.choice(
        [
            "Street",
            "Road",
            "Avenue",
            "Drive",
            "Court",
        ]
    )

    address = f"{street_number} {street_name} {street_type}"

    postcode = int(rng.integers(3000, 3999))

    return {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "phone": phone,
        "address": address,
        "postcode": postcode,
        "state": state if state is not None else "VIC",
    }


# ============================================================
# Base identity records
# ============================================================

def build_base_identity(customer_master):
    columns = detect_source_columns(customer_master)

    if columns["customer_id"] is None:
        raise ValueError(
            "Unable to identify customer ID column "
            "in customer_master.parquet"
        )

    rows = []

    for _, row in customer_master.iterrows():
        customer_id = str(row[columns["customer_id"]])

        state = (
            row[columns["state"]]
            if columns["state"]
            else None
        )

        fallback = build_fallback_identity(
            customer_id=customer_id,
            state=state,
        )

        record = {
            "true_person_id": customer_id,
            "customer_id": customer_id,
        }

        for field in [
            "first_name",
            "last_name",
            "email",
            "phone",
            "address",
            "postcode",
            "state",
        ]:
            source_col = columns[field]

            if source_col:
                value = row[source_col]

                if pd.isna(value) or str(value).strip() == "":
                    value = fallback[field]
            else:
                value = fallback[field]

            record[field] = value

        rows.append(record)

    return pd.DataFrame(rows)


# ============================================================
# Identity record generation
# ============================================================

def create_identity_source_record(
    base_row,
    record_number,
    source_system,
    variant_type,
    source_role,
):
    record = base_row.copy()

    record["identity_record_id"] = (
        f"IDR_{base_row['true_person_id']}_{record_number:02d}"
    )

    # Globally unique synthetic identity-record key.
    # Downstream operational transaction records will reference
    # this field rather than the hidden canonical customer ID.
    record["source_customer_id"] = record["identity_record_id"]

    record["source_system"] = source_system
    record["variant_type"] = variant_type
    record["source_role"] = source_role

    record["transaction_eligible"] = (
        source_system in TRANSACTION_CAPABLE_SYSTEMS
    )

    return record


def introduce_noise(record, severity="light"):
    record = record.copy()

    if severity == "light":
        mutation_count = random.choice([1, 1, 1, 2])

    elif severity == "medium":
        mutation_count = random.choice([2, 2, 3])

    else:
        mutation_count = random.choice([3, 4])

    possible_mutations = [
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "postcode",
        "missing_email",
        "missing_phone",
        "missing_address",
    ]

    selected = random.sample(
        possible_mutations,
        k=min(mutation_count, len(possible_mutations)),
    )

    for mutation in selected:
        if mutation == "first_name":
            record["first_name"] = alter_name(
                record["first_name"]
            )

        elif mutation == "last_name":
            record["last_name"] = alter_name(
                record["last_name"]
            )

        elif mutation == "email":
            record["email"] = alter_email(
                record["email"]
            )

        elif mutation == "phone":
            record["phone"] = alter_phone(
                record["phone"]
            )

        elif mutation == "address":
            record["address"] = alter_address(
                record["address"]
            )

        elif mutation == "postcode":
            if random.random() < 0.5:
                record["postcode"] = None

        elif mutation == "missing_email":
            record["email"] = None

        elif mutation == "missing_phone":
            record["phone"] = None

        elif mutation == "missing_address":
            record["address"] = None

    return record


def choose_additional_source_system(
    primary_source,
    used_sources,
):
    """
    Prefer cross-system diversity for duplicate identities.

    We do not require every customer to exist in every system.
    Instead, each person has one transaction-capable primary
    identity and may have additional identities elsewhere.
    """

    available_unused = [
        system
        for system in ALL_SOURCE_SYSTEMS
        if system not in used_sources
    ]

    if available_unused:
        # Give transaction-capable systems slightly more weight
        # because customer fragmentation often occurs between
        # POS, ecommerce and loyalty/account environments.
        weights = [
            2.0
            if system in TRANSACTION_CAPABLE_SYSTEMS
            else 1.0
            for system in available_unused
        ]

        return random.choices(
            available_unused,
            weights=weights,
            k=1,
        )[0]

    # More records than source systems: allow another identity
    # within an existing source environment.
    return random.choice(
        ALL_SOURCE_SYSTEMS
    )


def generate_identity_records(base_identity):
    """
    Create a realistic fragmented identity estate.

    Design:
    - every true person receives exactly one canonical source record
    - that canonical record is guaranteed to be transaction-capable
    - a subset of people receive additional noisy records
    - additional records are preferentially spread across source systems
    - Service / Marketing can enrich the identity graph but are not
      independently transaction-owning systems

    The resolver must infer identity from observable attributes only.
    true_person_id remains hidden ground truth for QA.
    """

    records = []

    for _, row in base_identity.iterrows():
        base = row.to_dict()

        draw = random.random()

        if draw < 0.64:
            record_count = 1

        elif draw < 0.87:
            record_count = 2

        elif draw < 0.96:
            record_count = 3

        elif draw < 0.99:
            record_count = 4

        else:
            record_count = 5

        primary_source = random.choices(
            TRANSACTION_CAPABLE_SYSTEMS,
            weights=PRIMARY_SOURCE_WEIGHTS,
            k=1,
        )[0]

        used_sources = {
            primary_source
        }

        # ----------------------------------------------------
        # Canonical operational source identity
        # ----------------------------------------------------

        records.append(
            create_identity_source_record(
                base_row=base,
                record_number=1,
                source_system=primary_source,
                variant_type="Canonical",
                source_role="Primary Transaction Identity",
            )
        )

        # ----------------------------------------------------
        # Additional fragmented / noisy identities
        # ----------------------------------------------------

        for record_number in range(
            2,
            record_count + 1,
        ):
            source_system = (
                choose_additional_source_system(
                    primary_source=primary_source,
                    used_sources=used_sources,
                )
            )

            used_sources.add(
                source_system
            )

            severity = random.choices(
                ["light", "medium", "heavy"],
                weights=[0.60, 0.30, 0.10],
                k=1,
            )[0]

            variant_type = (
                f"Synthetic {severity.title()} Variant"
            )

            noisy = introduce_noise(
                base,
                severity=severity,
            )

            source_role = (
                "Additional Transaction Identity"
                if source_system
                in TRANSACTION_CAPABLE_SYSTEMS
                else "Supporting Identity"
            )

            records.append(
                create_identity_source_record(
                    base_row=noisy,
                    record_number=record_number,
                    source_system=source_system,
                    variant_type=variant_type,
                    source_role=source_role,
                )
            )

    return pd.DataFrame(
        records
    )


# ============================================================
# Ambiguous household cases
# ============================================================

def create_ambiguous_households(identity_records):
    """
    Deliberately create shared household attributes among
    distinct people.

    These cases are valuable because the future resolver should
    avoid falsely merging customers solely because they share an
    address or phone.
    """

    df = identity_records.copy()

    canonical = (
        df[df["variant_type"] == "Canonical"]
        .drop_duplicates("true_person_id")
        .copy()
    )

    eligible = canonical[
        canonical["address"].notna()
        & canonical["postcode"].notna()
    ].copy()

    household_count = min(
        500,
        max(100, int(len(eligible) * 0.025)),
    )

    selected_ids = random.sample(
        eligible["true_person_id"].tolist(),
        k=min(household_count * 2, len(eligible)),
    )

    pairs = [
        selected_ids[i : i + 2]
        for i in range(0, len(selected_ids) - 1, 2)
    ]

    for pair_number, pair in enumerate(
        pairs[:household_count],
        start=1,
    ):
        person_a, person_b = pair

        source = canonical[
            canonical["true_person_id"] == person_a
        ].iloc[0]

        mask_b = df["true_person_id"] == person_b

        # Shared household address.
        df.loc[mask_b, "address"] = source["address"]
        df.loc[mask_b, "postcode"] = source["postcode"]

        df.loc[
            mask_b,
            "variant_type",
        ] = (
            df.loc[mask_b, "variant_type"].astype(str)
            + " | Shared Household"
        )

        df.loc[mask_b, "household_case_id"] = (
            f"HH_{pair_number:04d}"
        )

        mask_a = df["true_person_id"] == person_a

        df.loc[mask_a, "household_case_id"] = (
            f"HH_{pair_number:04d}"
        )

    return df


# ============================================================
# Ground truth
# ============================================================

def build_ground_truth(identity_records):
    return (
        identity_records[
            [
                "identity_record_id",
                "true_person_id",
            ]
        ]
        .copy()
        .rename(
            columns={
                "true_person_id": "golden_customer_id"
            }
        )
    )


# ============================================================
# QA Summary
# ============================================================

def print_summary(identity_records, ground_truth):
    print()
    print("=" * 68)
    print("IDENTITY RECORD GENERATION COMPLETE")
    print("=" * 68)

    print(
        f"Golden customers              : "
        f"{ground_truth['golden_customer_id'].nunique():,}"
    )

    print(
        f"Identity source records       : "
        f"{len(identity_records):,}"
    )

    average_records = (
        len(identity_records)
        / ground_truth["golden_customer_id"].nunique()
    )

    print(
        f"Average records / customer    : "
        f"{average_records:.2f}"
    )

    duplicate_people = (
        identity_records
        .groupby("true_person_id")
        .size()
        .gt(1)
        .sum()
    )

    print(
        f"Customers with duplicates     : "
        f"{duplicate_people:,}"
    )

    print(
        f"Duplicate customer rate       : "
        f"{duplicate_people / ground_truth['golden_customer_id'].nunique():.1%}"
    )

    ambiguous_households = (
        identity_records["household_case_id"]
        .notna()
        .sum()
        if "household_case_id" in identity_records.columns
        else 0
    )

    print(
        f"Records in household cases    : "
        f"{ambiguous_households:,}"
    )

    for field in [
        "email",
        "phone",
        "address",
    ]:
        missing_rate = (
            identity_records[field]
            .isna()
            .mean()
        )

        print(
            f"Missing {field:<20}: "
            f"{missing_rate:.1%}"
        )

    print()

    transaction_coverage = (
        identity_records
        .groupby("true_person_id")["transaction_eligible"]
        .any()
    )

    print(
        f"Customers with transaction ID: "
        f"{transaction_coverage.sum():,}"
    )

    print(
        f"Transaction identity coverage : "
        f"{transaction_coverage.mean():.1%}"
    )

    primary_counts = (
        identity_records[
            "source_role"
        ]
        .eq(
            "Primary Transaction Identity"
        )
        .groupby(
            identity_records[
                "true_person_id"
            ]
        )
        .sum()
    )

    print(
        f"Customers with one primary ID : "
        f"{primary_counts.eq(1).sum():,}"
    )

    print()
    print("Source-system distribution")
    print("-" * 68)

    print(
        identity_records[
            "source_system"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Source-role distribution")
    print("-" * 68)

    print(
        identity_records[
            "source_role"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("Variant distribution")
    print("-" * 68)

    print(
        identity_records[
            "variant_type"
        ]
        .value_counts()
        .head(15)
        .to_string()
    )

    print()
    print(
        f"Output: {OUTPUT_IDENTITY_PATH}"
    )

    print(
        f"Ground truth: {OUTPUT_GROUND_TRUTH_PATH}"
    )

    print("=" * 68)


# ============================================================
# Main
# ============================================================

def main():
    if not CUSTOMER_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Customer master not found: "
            f"{CUSTOMER_MASTER_PATH}"
        )

    customer_master = pd.read_parquet(
        CUSTOMER_MASTER_PATH
    )

    print(
        f"Loaded customer master: "
        f"{len(customer_master):,} customers"
    )

    base_identity = build_base_identity(
        customer_master
    )

    identity_records = generate_identity_records(
        base_identity
    )

    identity_records["household_case_id"] = None

    identity_records = create_ambiguous_households(
        identity_records
    )

    ground_truth = build_ground_truth(
        identity_records
    )

    OUTPUT_IDENTITY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    identity_records.to_parquet(
        OUTPUT_IDENTITY_PATH,
        index=False,
    )

    ground_truth.to_parquet(
        OUTPUT_GROUND_TRUTH_PATH,
        index=False,
    )

    print_summary(
        identity_records,
        ground_truth,
    )


if __name__ == "__main__":
    main()