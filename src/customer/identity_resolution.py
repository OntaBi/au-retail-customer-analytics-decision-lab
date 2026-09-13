from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IDENTITY_RECORDS_PATH = (
    ROOT / "data" / "generated" / "customer_identity_records.parquet"
)

GROUND_TRUTH_PATH = (
    ROOT / "data" / "generated" / "identity_ground_truth.parquet"
)

OUTPUT_RESOLUTION_PATH = (
    ROOT / "data" / "runtime" / "customer_identity_resolution.parquet"
)

OUTPUT_SUMMARY_PATH = (
    ROOT / "data" / "runtime" / "identity_resolution_summary.parquet"
)

OUTPUT_EDGE_AUDIT_PATH = (
    ROOT / "data" / "runtime" / "identity_edge_audit.parquet"
)


# ============================================================
# Thresholds
# ============================================================

PROBABILISTIC_THRESHOLD = 0.90
HIGH_CONFIDENCE_THRESHOLD = 0.96

CLUSTER_PAIR_THRESHOLD = 0.78

MAX_BLOCK_SIZE = 120
MAX_CLUSTER_SIZE_FOR_FULL_CHECK = 12


# ============================================================
# Union Find
# ============================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.members = {
            i: {i}
            for i in range(n)
        }

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )
        return self.parent[x]

    def get_members(self, x: int) -> set[int]:
        root = self.find(x)
        return self.members[root]

    def union(self, a: int, b: int) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        self.members[root_a] = (
            self.members[root_a]
            | self.members[root_b]
        )

        del self.members[root_b]

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return True


# ============================================================
# Normalisation
# ============================================================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"[^a-z0-9\s]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalise_name(value) -> str:
    return clean_text(value)


def normalise_email(value) -> str:
    if pd.isna(value):
        return ""

    value = (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
    )

    if "@" not in value:
        return value

    local, domain = value.split("@", 1)

    local = local.split("+", 1)[0]

    # Synthetic generator may insert dots.
    local = local.replace(".", "")

    return f"{local}@{domain}"


def normalise_phone(value) -> str:
    if pd.isna(value):
        return ""

    digits = re.sub(
        r"\D",
        "",
        str(value),
    )

    if not digits:
        return ""

    if (
        digits.startswith("61")
        and len(digits) >= 11
    ):
        digits = "0" + digits[2:]

    return digits


def normalise_address(value) -> str:
    if pd.isna(value):
        return ""

    value = (
        str(value)
        .lower()
        .strip()
    )

    replacements = {
        r"\bstreet\b": "st",
        r"\broad\b": "rd",
        r"\bavenue\b": "ave",
        r"\bdrive\b": "dr",
        r"\bcourt\b": "ct",
        r"\bboulevard\b": "blvd",
        r"\blane\b": "ln",
        r"\bplace\b": "pl",
    }

    for pattern, replacement in replacements.items():
        value = re.sub(
            pattern,
            replacement,
            value,
        )

    value = re.sub(
        r"[^a-z0-9\s]",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalise_postcode(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return re.sub(
        r"\D",
        "",
        value,
    )


def add_normalised_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    result["norm_first_name"] = (
        result["first_name"]
        .apply(normalise_name)
    )

    result["norm_last_name"] = (
        result["last_name"]
        .apply(normalise_name)
    )

    result["norm_email"] = (
        result["email"]
        .apply(normalise_email)
    )

    result["norm_phone"] = (
        result["phone"]
        .apply(normalise_phone)
    )

    result["norm_address"] = (
        result["address"]
        .apply(normalise_address)
    )

    result["norm_postcode"] = (
        result["postcode"]
        .apply(normalise_postcode)
    )

    result["norm_full_name"] = (
        result["norm_first_name"]
        + " "
        + result["norm_last_name"]
    ).str.strip()

    return result


# ============================================================
# Similarity
# ============================================================

def similarity(
    a: str,
    b: str,
) -> float:

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def phone_similarity(
    a: str,
    b: str,
) -> float:

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    if (
        len(a) >= 8
        and len(b) >= 8
        and a[-8:] == b[-8:]
    ):
        return 0.96

    return similarity(a, b)


def pair_match_score(
    row_a: pd.Series,
    row_b: pd.Series,
) -> tuple[float, dict]:

    components = {
        "first_name": similarity(
            row_a["norm_first_name"],
            row_b["norm_first_name"],
        ),
        "last_name": similarity(
            row_a["norm_last_name"],
            row_b["norm_last_name"],
        ),
        "email": similarity(
            row_a["norm_email"],
            row_b["norm_email"],
        ),
        "phone": phone_similarity(
            row_a["norm_phone"],
            row_b["norm_phone"],
        ),
        "address": similarity(
            row_a["norm_address"],
            row_b["norm_address"],
        ),
        "postcode": (
            1.0
            if (
                row_a["norm_postcode"]
                and row_a["norm_postcode"]
                == row_b["norm_postcode"]
            )
            else 0.0
        ),
    }

    weights = {
        "email": 0.32,
        "phone": 0.28,
        "last_name": 0.14,
        "first_name": 0.10,
        "address": 0.11,
        "postcode": 0.05,
    }

    weighted_score = 0.0
    available_weight = 0.0

    for field, weight in weights.items():

        if field == "postcode":
            available = bool(
                row_a["norm_postcode"]
                and row_b["norm_postcode"]
            )

        else:
            col = f"norm_{field}"

            available = bool(
                row_a[col]
                and row_b[col]
            )

        if available:
            weighted_score += (
                components[field]
                * weight
            )
            available_weight += weight

    if available_weight == 0:
        return 0.0, components

    return (
        weighted_score
        / available_weight,
        components,
    )


# ============================================================
# Evidence rules
# ============================================================

def strong_email_match(
    components: dict,
) -> bool:
    return components["email"] >= 0.97


def strong_phone_match(
    components: dict,
) -> bool:
    return components["phone"] >= 0.97


def strong_name_match(
    components: dict,
) -> bool:
    return (
        components["first_name"] >= 0.82
        and components["last_name"] >= 0.85
    )


def supporting_address_match(
    components: dict,
) -> bool:
    return (
        components["address"] >= 0.88
        and components["postcode"] >= 1.0
    )


def count_identity_signals(
    components: dict,
) -> int:

    signals = 0

    if strong_email_match(components):
        signals += 1

    if strong_phone_match(components):
        signals += 1

    if strong_name_match(components):
        signals += 1

    if supporting_address_match(components):
        signals += 1

    return signals


def has_material_conflict(
    components: dict,
) -> bool:

    severe_name_conflict = (
        components["first_name"] < 0.30
        and components["last_name"] < 0.30
    )

    severe_email_conflict = (
        components["email"] > 0
        and components["email"] < 0.35
    )

    severe_phone_conflict = (
        components["phone"] > 0
        and components["phone"] < 0.35
    )

    # Strong email OR strong phone can override
    # a name typo / initial issue.
    if (
        severe_name_conflict
        and not strong_email_match(components)
        and not strong_phone_match(components)
    ):
        return True

    # If both email and phone are present and
    # materially disagree, require stronger corroboration.
    if (
        severe_email_conflict
        and severe_phone_conflict
        and not strong_name_match(components)
    ):
        return True

    return False


# ============================================================
# Edge audit
# ============================================================

def make_edge_audit_row(
    df: pd.DataFrame,
    idx_a: int,
    idx_b: int,
    method: str,
    score: float,
    accepted: bool,
    reason: str,
    components: dict | None = None,
) -> dict:

    row = {
        "record_a": df.loc[
            idx_a,
            "identity_record_id",
        ],
        "record_b": df.loc[
            idx_b,
            "identity_record_id",
        ],
        "match_method": method,
        "score": score,
        "accepted": accepted,
        "rejection_reason": reason,
    }

    if components:
        for field, value in components.items():
            row[f"{field}_similarity"] = value

    return row


# ============================================================
# Cluster consistency
# ============================================================

def cluster_consistency_check(
    df: pd.DataFrame,
    uf: UnionFind,
    idx_a: int,
    idx_b: int,
) -> tuple[bool, str]:

    members_a = list(
        uf.get_members(idx_a)
    )

    members_b = list(
        uf.get_members(idx_b)
    )

    # Keep check computationally manageable.
    if (
        len(members_a)
        > MAX_CLUSTER_SIZE_FOR_FULL_CHECK
    ):
        members_a = members_a[
            :MAX_CLUSTER_SIZE_FOR_FULL_CHECK
        ]

    if (
        len(members_b)
        > MAX_CLUSTER_SIZE_FOR_FULL_CHECK
    ):
        members_b = members_b[
            :MAX_CLUSTER_SIZE_FOR_FULL_CHECK
        ]

    scores = []
    severe_conflicts = 0

    for member_a in members_a:
        for member_b in members_b:

            score, components = (
                pair_match_score(
                    df.loc[member_a],
                    df.loc[member_b],
                )
            )

            scores.append(score)

            if has_material_conflict(
                components
            ):
                severe_conflicts += 1

    if not scores:
        return True, "No prior cluster evidence"

    mean_score = np.mean(scores)
    max_score = np.max(scores)

    comparisons = len(scores)

    conflict_rate = (
        severe_conflicts
        / comparisons
    )

    # Never join clusters where a substantial portion
    # of cross-cluster identities materially disagree.
    if conflict_rate > 0.25:
        return (
            False,
            "Cluster conflict rate too high",
        )

    # For larger clusters we require broad consistency,
    # not one exceptionally strong bridge.
    if (
        len(members_a) > 1
        or len(members_b) > 1
    ):
        if (
            mean_score
            < CLUSTER_PAIR_THRESHOLD
            and max_score < 0.96
        ):
            return (
                False,
                "Insufficient cluster-wide consistency",
            )

    return (
        True,
        "Cluster consistency passed",
    )


# ============================================================
# Match registration
# ============================================================

def update_record_match(
    record_match: dict,
    index: int,
    method: str,
    confidence: float,
):

    existing = record_match.get(index)

    if (
        existing is None
        or confidence
        > existing["confidence"]
    ):
        record_match[index] = {
            "method": method,
            "confidence": confidence,
        }


def attempt_link(
    df: pd.DataFrame,
    uf: UnionFind,
    record_match: dict,
    edge_audit: list,
    idx_a: int,
    idx_b: int,
    method: str,
    confidence: float,
    components: dict,
    apply_cluster_gate: bool = True,
) -> bool:

    if uf.find(idx_a) == uf.find(idx_b):
        return False

    if apply_cluster_gate:
        cluster_ok, reason = (
            cluster_consistency_check(
                df,
                uf,
                idx_a,
                idx_b,
            )
        )

        if not cluster_ok:
            edge_audit.append(
                make_edge_audit_row(
                    df,
                    idx_a,
                    idx_b,
                    method,
                    confidence,
                    False,
                    reason,
                    components,
                )
            )

            return False

    uf.union(
        idx_a,
        idx_b,
    )

    update_record_match(
        record_match,
        idx_a,
        method,
        confidence,
    )

    update_record_match(
        record_match,
        idx_b,
        method,
        confidence,
    )

    edge_audit.append(
        make_edge_audit_row(
            df,
            idx_a,
            idx_b,
            method,
            confidence,
            True,
            "Accepted",
            components,
        )
    )

    return True


# ============================================================
# Strong deterministic matching
# ============================================================

def exact_group_pairs(
    df: pd.DataFrame,
    field: str,
):
    valid = df[
        df[field] != ""
    ]

    for _, group in valid.groupby(
        field
    ):
        if len(group) < 2:
            continue

        indices = group.index.tolist()

        for pair in combinations(
            indices,
            2,
        ):
            yield pair


def deterministic_resolution(
    df: pd.DataFrame,
    uf: UnionFind,
    record_match: dict,
    edge_audit: list,
):

    # --------------------------------------------------------
    # Exact email
    # --------------------------------------------------------

    for idx_a, idx_b in exact_group_pairs(
        df,
        "norm_email",
    ):
        _, components = (
            pair_match_score(
                df.loc[idx_a],
                df.loc[idx_b],
            )
        )

        # Exact email remains strong, but cluster gate
        # protects against transitive contamination.
        attempt_link(
            df,
            uf,
            record_match,
            edge_audit,
            idx_a,
            idx_b,
            "Exact Email",
            0.99,
            components,
            apply_cluster_gate=True,
        )

    # --------------------------------------------------------
    # Exact phone
    # --------------------------------------------------------

    for idx_a, idx_b in exact_group_pairs(
        df,
        "norm_phone",
    ):
        _, components = (
            pair_match_score(
                df.loc[idx_a],
                df.loc[idx_b],
            )
        )

        attempt_link(
            df,
            uf,
            record_match,
            edge_audit,
            idx_a,
            idx_b,
            "Exact Phone",
            0.99,
            components,
            apply_cluster_gate=True,
        )


# ============================================================
# Candidate generation
# ============================================================

def first_char(
    value: str,
) -> str:
    return (
        value[:1]
        if value
        else ""
    )


def prefix(
    value: str,
    n: int,
) -> str:
    return (
        value[:n]
        if value
        else ""
    )


def email_parts(
    value: str,
) -> tuple[str, str]:

    if not value or "@" not in value:
        return "", ""

    return tuple(
        value.split("@", 1)
    )


def house_number(
    address: str,
) -> str:

    if not address:
        return ""

    match = re.match(
        r"(\d+)",
        address,
    )

    return (
        match.group(1)
        if match
        else ""
    )


def create_candidate_pairs(
    df: pd.DataFrame,
) -> list[tuple[int, int]]:

    blocks = defaultdict(list)

    for idx, row in df.iterrows():

        postcode = row[
            "norm_postcode"
        ]

        first_name = row[
            "norm_first_name"
        ]

        last_name = row[
            "norm_last_name"
        ]

        email = row[
            "norm_email"
        ]

        phone = row[
            "norm_phone"
        ]

        address = row[
            "norm_address"
        ]

        # Name + postcode is now ONLY a candidate block,
        # never an automatic match.
        if (
            postcode
            and first_name
            and last_name
        ):
            blocks[
                (
                    "NP",
                    postcode,
                    first_char(first_name),
                    prefix(last_name, 3),
                )
            ].append(idx)

        if (
            postcode
            and last_name
        ):
            blocks[
                (
                    "LP",
                    postcode,
                    prefix(last_name, 4),
                )
            ].append(idx)

        local, domain = (
            email_parts(email)
        )

        if local and domain:
            blocks[
                (
                    "EM",
                    prefix(local, 4),
                    domain,
                )
            ].append(idx)

        if len(phone) >= 8:
            blocks[
                (
                    "PH",
                    phone[-8:],
                )
            ].append(idx)

        number = house_number(
            address
        )

        if (
            number
            and postcode
            and last_name
        ):
            blocks[
                (
                    "AD",
                    number,
                    postcode,
                    first_char(last_name),
                )
            ].append(idx)

    candidate_pairs = set()

    for members in blocks.values():

        members = list(
            dict.fromkeys(members)
        )

        if len(members) < 2:
            continue

        if len(members) > MAX_BLOCK_SIZE:
            continue

        for pair in combinations(
            sorted(members),
            2,
        ):
            candidate_pairs.add(
                pair
            )

    return list(
        candidate_pairs
    )


# ============================================================
# Corroborated probabilistic matching
# ============================================================

def passes_probabilistic_rules(
    score: float,
    components: dict,
) -> tuple[bool, str]:

    if score < PROBABILISTIC_THRESHOLD:
        return (
            False,
            "Below probabilistic threshold",
        )

    if has_material_conflict(
        components
    ):
        return (
            False,
            "Material identity conflict",
        )

    signal_count = (
        count_identity_signals(
            components
        )
    )

    # Core v2 rule:
    # one strong clue is not enough.
    if signal_count < 2:
        return (
            False,
            "Insufficient corroborating signals",
        )

    # Address + postcode without strong person-level
    # identifiers must never resolve an identity.
    person_level_signal = (
        strong_email_match(components)
        or strong_phone_match(components)
        or strong_name_match(components)
    )

    if not person_level_signal:
        return (
            False,
            "No person-level identity signal",
        )

    # If only name + address support the match,
    # require extremely strong name similarity.
    if (
        strong_name_match(components)
        and supporting_address_match(
            components
        )
        and not strong_email_match(
            components
        )
        and not strong_phone_match(
            components
        )
    ):
        if (
            components["first_name"]
            < 0.94
            or components["last_name"]
            < 0.94
        ):
            return (
                False,
                "Name/address evidence not strong enough",
            )

    return (
        True,
        "Probabilistic criteria passed",
    )


def probabilistic_resolution(
    df: pd.DataFrame,
    uf: UnionFind,
    record_match: dict,
    edge_audit: list,
) -> tuple[int, int]:

    candidate_pairs = (
        create_candidate_pairs(df)
    )

    accepted = 0

    # Evaluate strongest pairs first to create
    # stable high-confidence clusters before weaker ones.
    scored_pairs = []

    for idx_a, idx_b in candidate_pairs:

        if uf.find(idx_a) == uf.find(idx_b):
            continue

        score, components = (
            pair_match_score(
                df.loc[idx_a],
                df.loc[idx_b],
            )
        )

        scored_pairs.append(
            (
                score,
                idx_a,
                idx_b,
                components,
            )
        )

    scored_pairs.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    for (
        score,
        idx_a,
        idx_b,
        components,
    ) in scored_pairs:

        if uf.find(idx_a) == uf.find(idx_b):
            continue

        passed, reason = (
            passes_probabilistic_rules(
                score,
                components,
            )
        )

        if not passed:

            edge_audit.append(
                make_edge_audit_row(
                    df,
                    idx_a,
                    idx_b,
                    "Probabilistic",
                    score,
                    False,
                    reason,
                    components,
                )
            )

            continue

        if (
            score
            >= HIGH_CONFIDENCE_THRESHOLD
        ):
            method = (
                "Probabilistic High Confidence"
            )
        else:
            method = "Probabilistic"

        linked = attempt_link(
            df,
            uf,
            record_match,
            edge_audit,
            idx_a,
            idx_b,
            method,
            score,
            components,
            apply_cluster_gate=True,
        )

        if linked:
            accepted += 1

    return (
        len(candidate_pairs),
        accepted,
    )


# ============================================================
# Output construction
# ============================================================

def build_resolution_output(
    df: pd.DataFrame,
    uf: UnionFind,
    record_match: dict,
) -> pd.DataFrame:

    result = df.copy()

    result["component_root"] = [
        uf.find(i)
        for i in range(len(result))
    ]

    component_members = (
        result
        .groupby(
            "component_root"
        )["identity_record_id"]
        .apply(list)
        .to_dict()
    )

    sorted_components = sorted(
        component_members.items(),
        key=lambda item: min(
            item[1]
        ),
    )

    root_to_golden = {}

    for counter, (
        root,
        _,
    ) in enumerate(
        sorted_components,
        start=1,
    ):
        root_to_golden[
            root
        ] = (
            f"GOLDEN_{counter:06d}"
        )

    result[
        "golden_customer_id"
    ] = (
        result["component_root"]
        .map(root_to_golden)
    )

    cluster_sizes = (
        result
        .groupby(
            "golden_customer_id"
        )
        .size()
        .rename(
            "records_linked"
        )
    )

    result = result.merge(
        cluster_sizes,
        on="golden_customer_id",
        how="left",
    )

    methods = []
    confidences = []

    for idx in result.index:

        match = (
            record_match.get(idx)
        )

        if match:
            methods.append(
                match["method"]
            )
            confidences.append(
                match["confidence"]
            )

        elif (
            result.loc[
                idx,
                "records_linked",
            ]
            == 1
        ):
            methods.append(
                "Standalone"
            )
            confidences.append(
                1.00
            )

        else:
            methods.append(
                "Cluster Link"
            )
            confidences.append(
                0.90
            )

    result["match_method"] = methods
    result["match_confidence"] = (
        confidences
    )

    result[
        "identity_resolution_status"
    ] = np.where(
        result["records_linked"] > 1,
        "Resolved",
        "Standalone",
    )

    result[
        "ambiguous_match_flag"
    ] = (
        result["match_confidence"]
        < 0.90
    )

    return result.drop(
        columns=["component_root"]
    )


# ============================================================
# Evaluation
# ============================================================

def create_pair_set(
    df: pd.DataFrame,
    group_col: str,
    id_col: str,
) -> set:

    pair_set = set()

    for _, group in df.groupby(
        group_col
    ):

        ids = sorted(
            group[id_col].tolist()
        )

        if len(ids) < 2:
            continue

        pair_set.update(
            combinations(ids, 2)
        )

    return pair_set


def evaluate_resolution(
    resolution: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:

    evaluation = (
        resolution[
            [
                "identity_record_id",
                "golden_customer_id",
            ]
        ]
        .merge(
            ground_truth.rename(
                columns={
                    "golden_customer_id":
                    "true_golden_customer_id"
                }
            ),
            on="identity_record_id",
            how="left",
            validate="one_to_one",
        )
    )

    true_pairs = (
        create_pair_set(
            evaluation,
            "true_golden_customer_id",
            "identity_record_id",
        )
    )

    predicted_pairs = (
        create_pair_set(
            evaluation,
            "golden_customer_id",
            "identity_record_id",
        )
    )

    true_positive = len(
        true_pairs
        & predicted_pairs
    )

    false_positive = len(
        predicted_pairs
        - true_pairs
    )

    false_negative = len(
        true_pairs
        - predicted_pairs
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        )
        else 1.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        )
        else 1.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    false_merge_rate = (
        false_positive
        / len(predicted_pairs)
        if predicted_pairs
        else 0.0
    )

    missed_link_rate = (
        false_negative
        / len(true_pairs)
        if true_pairs
        else 0.0
    )

    return {
        "true_pairs": len(
            true_pairs
        ),
        "predicted_pairs": len(
            predicted_pairs
        ),
        "true_positive_pairs": (
            true_positive
        ),
        "false_positive_pairs": (
            false_positive
        ),
        "false_negative_pairs": (
            false_negative
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_merge_rate": (
            false_merge_rate
        ),
        "missed_link_rate": (
            missed_link_rate
        ),
    }


# ============================================================
# Summary
# ============================================================

def build_summary(
    resolution: pd.DataFrame,
    metrics: dict,
    candidate_pairs: int,
    accepted_probabilistic: int,
    edge_audit: pd.DataFrame,
) -> pd.DataFrame:

    values = {
        "source_identity_records": (
            len(resolution)
        ),
        "predicted_golden_customers": (
            resolution[
                "golden_customer_id"
            ].nunique()
        ),
        "candidate_pairs_evaluated": (
            candidate_pairs
        ),
        "probabilistic_links_accepted": (
            accepted_probabilistic
        ),
        "accepted_edges": int(
            edge_audit[
                "accepted"
            ].sum()
        )
        if not edge_audit.empty
        else 0,
        "rejected_edges": int(
            (
                ~edge_audit[
                    "accepted"
                ]
            ).sum()
        )
        if not edge_audit.empty
        else 0,
        **metrics,
    }

    return pd.DataFrame(
        {
            "metric": list(
                values.keys()
            ),
            "value": list(
                values.values()
            ),
        }
    )


def print_summary(
    resolution: pd.DataFrame,
    metrics: dict,
    candidate_pairs: int,
    accepted_probabilistic: int,
    edge_audit: pd.DataFrame,
):

    print()
    print("=" * 74)
    print("IDENTITY RESOLUTION V2 COMPLETE")
    print("=" * 74)

    print(
        f"Source identity records       : "
        f"{len(resolution):,}"
    )

    print(
        f"Predicted golden customers    : "
        f"{resolution['golden_customer_id'].nunique():,}"
    )

    print(
        f"Candidate pairs evaluated     : "
        f"{candidate_pairs:,}"
    )

    print(
        f"Probabilistic links accepted  : "
        f"{accepted_probabilistic:,}"
    )

    if not edge_audit.empty:

        accepted_edges = int(
            edge_audit[
                "accepted"
            ].sum()
        )

        rejected_edges = int(
            (
                ~edge_audit[
                    "accepted"
                ]
            ).sum()
        )

        print(
            f"Accepted audit edges          : "
            f"{accepted_edges:,}"
        )

        print(
            f"Rejected audit edges          : "
            f"{rejected_edges:,}"
        )

    print()
    print("MATCH METHOD")
    print("-" * 74)

    print(
        resolution[
            "match_method"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("IDENTITY QUALITY")
    print("-" * 74)

    print(
        f"True identity pairs           : "
        f"{metrics['true_pairs']:,}"
    )

    print(
        f"Predicted identity pairs      : "
        f"{metrics['predicted_pairs']:,}"
    )

    print(
        f"True-positive links           : "
        f"{metrics['true_positive_pairs']:,}"
    )

    print(
        f"False merges                  : "
        f"{metrics['false_positive_pairs']:,}"
    )

    print(
        f"Missed links                  : "
        f"{metrics['false_negative_pairs']:,}"
    )

    print()

    print(
        f"Precision                     : "
        f"{metrics['precision']:.2%}"
    )

    print(
        f"Recall                        : "
        f"{metrics['recall']:.2%}"
    )

    print(
        f"F1                            : "
        f"{metrics['f1']:.2%}"
    )

    print(
        f"False merge rate              : "
        f"{metrics['false_merge_rate']:.2%}"
    )

    print(
        f"Missed link rate              : "
        f"{metrics['missed_link_rate']:.2%}"
    )

    print()
    print("EDGE REJECTION REASONS")
    print("-" * 74)

    if edge_audit.empty:
        print("No edge audit records.")

    else:
        rejected = edge_audit[
            ~edge_audit["accepted"]
        ]

        if rejected.empty:
            print("No rejected edges.")

        else:
            print(
                rejected[
                    "rejection_reason"
                ]
                .value_counts()
                .head(15)
                .to_string()
            )

    print()
    print(
        f"Resolution output             : "
        f"{OUTPUT_RESOLUTION_PATH}"
    )

    print(
        f"Summary output                : "
        f"{OUTPUT_SUMMARY_PATH}"
    )

    print(
        f"Edge audit                    : "
        f"{OUTPUT_EDGE_AUDIT_PATH}"
    )

    print("=" * 74)


# ============================================================
# Main
# ============================================================

def main():

    if not IDENTITY_RECORDS_PATH.exists():
        raise FileNotFoundError(
            f"Identity records not found: "
            f"{IDENTITY_RECORDS_PATH}"
        )

    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth not found: "
            f"{GROUND_TRUTH_PATH}"
        )

    source = pd.read_parquet(
        IDENTITY_RECORDS_PATH
    )

    ground_truth = pd.read_parquet(
        GROUND_TRUTH_PATH
    )

    print(
        f"Loaded identity source records: "
        f"{len(source):,}"
    )

    # ========================================================
    # Leakage protection
    # ========================================================

    prohibited_fields = [
        "true_person_id",
        "customer_id",
        "variant_type",
        "household_case_id",
    ]

    resolver_source = (
        source.drop(
            columns=[
                c
                for c in prohibited_fields
                if c in source.columns
            ],
            errors="ignore",
        )
        .reset_index(drop=True)
    )

    resolver_source = (
        add_normalised_fields(
            resolver_source
        )
    )

    uf = UnionFind(
        len(resolver_source)
    )

    record_match = {}
    edge_audit = []

    # ========================================================
    # Pass 1 — Strong deterministic
    # ========================================================

    print(
        "Running strong deterministic matching..."
    )

    deterministic_resolution(
        resolver_source,
        uf,
        record_match,
        edge_audit,
    )

    # ========================================================
    # Pass 2 — Corroborated probabilistic
    # ========================================================

    print(
        "Building probabilistic candidate pairs..."
    )

    (
        candidate_pairs,
        accepted_probabilistic,
    ) = probabilistic_resolution(
        resolver_source,
        uf,
        record_match,
        edge_audit,
    )

    print(
        f"Evaluated {candidate_pairs:,} candidate pairs"
    )

    # ========================================================
    # Resolution output
    # ========================================================

    resolution = (
        build_resolution_output(
            resolver_source,
            uf,
            record_match,
        )
    )

    output_columns = [
        "identity_record_id",
        "source_system",
        "first_name",
        "last_name",
        "email",
        "phone",
        "address",
        "postcode",
        "state",
        "golden_customer_id",
        "records_linked",
        "match_method",
        "match_confidence",
        "identity_resolution_status",
        "ambiguous_match_flag",
    ]

    resolution = resolution[
        [
            c
            for c in output_columns
            if c in resolution.columns
        ]
    ].copy()

    # ========================================================
    # QA only — ground truth enters here
    # ========================================================

    metrics = evaluate_resolution(
        resolution,
        ground_truth,
    )

    edge_audit_df = pd.DataFrame(
        edge_audit
    )

    summary = build_summary(
        resolution,
        metrics,
        candidate_pairs,
        accepted_probabilistic,
        edge_audit_df,
    )

    OUTPUT_RESOLUTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    resolution.to_parquet(
        OUTPUT_RESOLUTION_PATH,
        index=False,
    )

    summary.to_parquet(
        OUTPUT_SUMMARY_PATH,
        index=False,
    )

    edge_audit_df.to_parquet(
        OUTPUT_EDGE_AUDIT_PATH,
        index=False,
    )

    print_summary(
        resolution,
        metrics,
        candidate_pairs,
        accepted_probabilistic,
        edge_audit_df,
    )


if __name__ == "__main__":
    main()