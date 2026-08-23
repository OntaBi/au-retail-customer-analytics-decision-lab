from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CUSTOMER_MASTER_FILE = Path(
    "data/generated/customer_master.parquet"
)

TRANSACTION_FILE = Path(
    "data/generated/transactions.parquet"
)

CLUSTER_FILE = Path(
    "data/runtime/customer_clusters.parquet"
)

OUTPUT_DIR = Path("outputs")

CLUSTER_PROFILE_FILE = (
    OUTPUT_DIR / "customer_cluster_profile.csv"
)

PERSONA_BY_CLUSTER_FILE = (
    OUTPUT_DIR / "persona_by_cluster.csv"
)

CLUSTER_BY_PERSONA_FILE = (
    OUTPUT_DIR / "cluster_by_persona.csv"
)

ARCHETYPE_BY_CLUSTER_FILE = (
    OUTPUT_DIR / "archetype_by_cluster.csv"
)


# ---------------------------------------------------------------------
# Category profile
# ---------------------------------------------------------------------

def build_category_profile(
    transactions: pd.DataFrame,
) -> pd.DataFrame:

    category_counts = (
        transactions
        .groupby(
            [
                "customer_id",
                "category",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    total_orders = (
        category_counts.sum(axis=1)
    )

    category_share = (
        category_counts
        .div(
            total_orders,
            axis=0,
        )
    )

    category_share.columns = [
        "category_share_"
        + column.lower()
        .replace(" & ", "_")
        .replace(" ", "_")
        for column in category_share.columns
    ]

    category_share = (
        category_share
        .reset_index()
    )

    dominant_category = (
        category_counts
        .idxmax(axis=1)
        .rename(
            "dominant_category"
        )
        .reset_index()
    )

    result = category_share.merge(
        dominant_category,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    return result


# ---------------------------------------------------------------------
# Build validation dataset
# ---------------------------------------------------------------------

def build_validation_data(
    customer_master: pd.DataFrame,
    transactions: pd.DataFrame,
    clustered: pd.DataFrame,
) -> pd.DataFrame:

    truth = customer_master[
        [
            "customer_id",
            "shopping_persona",
            "archetype",
        ]
    ].copy()

    category_profile = (
        build_category_profile(
            transactions
        )
    )

    validation = (
        clustered
        .merge(
            truth,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            category_profile,
            on="customer_id",
            how="left",
            validate="one_to_one",
        )
    )

    return validation


# ---------------------------------------------------------------------
# Behavioural cluster profile
# ---------------------------------------------------------------------

def build_cluster_profile(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    profile = (
        validation
        .groupby(
            [
                "cluster",
                "cluster_name",
            ]
        )
        .agg(
            customers=(
                "customer_id",
                "nunique",
            ),
            median_orders=(
                "orders",
                "median",
            ),
            median_aov=(
                "avg_order_value",
                "median",
            ),
            median_margin_per_order=(
                "avg_margin_per_order",
                "median",
            ),
            median_units_per_order=(
                "units_per_order",
                "median",
            ),
            median_discount_pct=(
                "avg_discount_pct",
                "median",
            ),
            median_discounted_order_share=(
                "discounted_order_share",
                "median",
            ),
            median_store_share=(
                "store_share",
                "median",
            ),
            median_online_share=(
                "online_share",
                "median",
            ),
            median_click_collect_share=(
                "click_collect_share",
                "median",
            ),
            median_channels_used=(
                "channels_used",
                "median",
            ),
            median_categories_used=(
                "categories_used",
                "median",
            ),
            median_category_concentration=(
                "category_concentration",
                "median",
            ),
            median_dominant_category_share=(
                "dominant_category_share",
                "median",
            ),
            median_purchase_gap=(
                "median_purchase_gap_days",
                "median",
            ),
            median_cadence_cv=(
                "cadence_cv",
                "median",
            ),
        )
        .reset_index()
    )

    profile["cluster_share_pct"] = (
        profile["customers"]
        / profile["customers"].sum()
        * 100
    )

    numeric_columns = [
        column
        for column in profile.columns
        if column not in [
            "cluster",
            "customers",
        ]
    ]

    profile[numeric_columns] = (
        profile[numeric_columns]
        .round(3)
    )

    return profile


# ---------------------------------------------------------------------
# Category mix by cluster
# ---------------------------------------------------------------------

def build_cluster_category_mix(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    category_columns = [
        column
        for column in validation.columns
        if column.startswith(
            "category_share_"
        )
    ]

    category_mix = (
        validation
        .groupby("cluster")[
            category_columns
        ]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )

    return category_mix


# ---------------------------------------------------------------------
# Persona distribution within clusters
# ---------------------------------------------------------------------

def build_persona_by_cluster(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["cluster"],
        validation["shopping_persona"],
        normalize="index",
    )

    return (
        crosstab
        .mul(100)
        .round(1)
    )


# ---------------------------------------------------------------------
# Cluster distribution within personas
# ---------------------------------------------------------------------

def build_cluster_by_persona(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["shopping_persona"],
        validation["cluster"],
        normalize="index",
    )

    return (
        crosstab
        .mul(100)
        .round(1)
    )


# ---------------------------------------------------------------------
# Behavioural archetype distribution within clusters
# ---------------------------------------------------------------------

def build_archetype_by_cluster(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    crosstab = pd.crosstab(
        validation["cluster"],
        validation["archetype"],
        normalize="index",
    )

    return (
        crosstab
        .mul(100)
        .round(1)
    )


# ---------------------------------------------------------------------
# Cluster purity / enrichment
# ---------------------------------------------------------------------

def build_cluster_enrichment(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    overall_persona_share = (
        validation[
            "shopping_persona"
        ]
        .value_counts(
            normalize=True
        )
    )

    rows = []

    for cluster, group in (
        validation
        .groupby("cluster")
    ):

        persona_share = (
            group[
                "shopping_persona"
            ]
            .value_counts(
                normalize=True
            )
        )

        dominant_persona = (
            persona_share.index[0]
        )

        dominant_share = (
            persona_share.iloc[0]
        )

        baseline_share = (
            overall_persona_share[
                dominant_persona
            ]
        )

        enrichment = (
            dominant_share
            / baseline_share
        )

        rows.append(
            {
                "cluster":
                    cluster,
                "cluster_name":
                    group[
                        "cluster_name"
                    ].iloc[0],
                "dominant_persona":
                    dominant_persona,
                "dominant_persona_share_pct":
                    dominant_share * 100,
                "baseline_persona_share_pct":
                    baseline_share * 100,
                "persona_enrichment":
                    enrichment,
            }
        )

    enrichment = pd.DataFrame(
        rows
    )

    enrichment[
        [
            "dominant_persona_share_pct",
            "baseline_persona_share_pct",
            "persona_enrichment",
        ]
    ] = (
        enrichment[
            [
                "dominant_persona_share_pct",
                "baseline_persona_share_pct",
                "persona_enrichment",
            ]
        ]
        .round(2)
    )

    return enrichment


# ---------------------------------------------------------------------
# Dominant observed category
# ---------------------------------------------------------------------

def build_dominant_category_summary(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    category_summary = (
        validation
        .groupby("cluster")[
            "dominant_category"
        ]
        .agg(
            lambda values:
                values.value_counts()
                .index[0]
        )
        .reset_index(
            name="most_common_dominant_category"
        )
    )

    category_share = (
        validation
        .groupby("cluster")[
            "dominant_category"
        ]
        .apply(
            lambda values:
                values.value_counts(
                    normalize=True
                ).iloc[0]
        )
        .mul(100)
        .round(1)
        .reset_index(
            name="dominant_category_customer_share_pct"
        )
    )

    return category_summary.merge(
        category_share,
        on="cluster",
        how="left",
    )


# ---------------------------------------------------------------------
# Print output
# ---------------------------------------------------------------------

def print_results(
    validation: pd.DataFrame,
    profile: pd.DataFrame,
    category_mix: pd.DataFrame,
    persona_by_cluster: pd.DataFrame,
    cluster_by_persona: pd.DataFrame,
    archetype_by_cluster: pd.DataFrame,
    enrichment: pd.DataFrame,
    dominant_category: pd.DataFrame,
) -> None:

    print(
        "\nCUSTOMER CLUSTER VALIDATION"
    )
    print("=" * 100)

    print(
        f"Customers validated: "
        f"{validation['customer_id'].nunique():,}"
    )

    print(
        "\nCLUSTER BEHAVIOURAL PROFILE"
    )
    print("-" * 100)

    print(
        profile.to_string(
            index=False
        )
    )

    print(
        "\nCATEGORY MIX BY CLUSTER (%)"
    )
    print("-" * 100)

    print(
        category_mix.to_string(
            index=False
        )
    )

    print(
        "\nDOMINANT CATEGORY BY CLUSTER"
    )
    print("-" * 100)

    print(
        dominant_category.to_string(
            index=False
        )
    )

    print(
        "\nHIDDEN PERSONA MIX WITHIN EACH CLUSTER (%)"
    )
    print("-" * 100)

    print(
        persona_by_cluster.to_string()
    )

    print(
        "\nCLUSTER DISTRIBUTION WITHIN EACH HIDDEN PERSONA (%)"
    )
    print("-" * 100)

    print(
        cluster_by_persona.to_string()
    )

    print(
        "\nBEHAVIOURAL ARCHETYPE MIX WITHIN EACH CLUSTER (%)"
    )
    print("-" * 100)

    print(
        archetype_by_cluster.to_string()
    )

    print(
        "\nDOMINANT PERSONA / ENRICHMENT"
    )
    print("-" * 100)

    print(
        enrichment.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading cluster validation data..."
    )

    customer_master = pd.read_parquet(
        CUSTOMER_MASTER_FILE
    )

    transactions = pd.read_parquet(
        TRANSACTION_FILE
    )

    clustered = pd.read_parquet(
        CLUSTER_FILE
    )

    validation = build_validation_data(
        customer_master,
        transactions,
        clustered,
    )

    profile = build_cluster_profile(
        validation
    )

    category_mix = (
        build_cluster_category_mix(
            validation
        )
    )

    persona_by_cluster = (
        build_persona_by_cluster(
            validation
        )
    )

    cluster_by_persona = (
        build_cluster_by_persona(
            validation
        )
    )

    archetype_by_cluster = (
        build_archetype_by_cluster(
            validation
        )
    )

    enrichment = (
        build_cluster_enrichment(
            validation
        )
    )

    dominant_category = (
        build_dominant_category_summary(
            validation
        )
    )

    profile.to_csv(
        CLUSTER_PROFILE_FILE,
        index=False,
    )

    persona_by_cluster.to_csv(
        PERSONA_BY_CLUSTER_FILE
    )

    cluster_by_persona.to_csv(
        CLUSTER_BY_PERSONA_FILE
    )

    archetype_by_cluster.to_csv(
        ARCHETYPE_BY_CLUSTER_FILE
    )

    print_results(
        validation,
        profile,
        category_mix,
        persona_by_cluster,
        cluster_by_persona,
        archetype_by_cluster,
        enrichment,
        dominant_category,
    )

    print(
        "\nFiles created:"
    )
    print(
        CLUSTER_PROFILE_FILE
    )
    print(
        PERSONA_BY_CLUSTER_FILE
    )
    print(
        CLUSTER_BY_PERSONA_FILE
    )
    print(
        ARCHETYPE_BY_CLUSTER_FILE
    )


if __name__ == "__main__":
    main()