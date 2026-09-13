from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "data/runtime/clustering_features.parquet"
)

OUTPUT_DIR = Path("outputs")

DIAGNOSTICS_FILE = (
    OUTPUT_DIR / "clustering_diagnostics.csv"
)

CLUSTERED_FILE = Path(
    "data/runtime/customer_clusters.parquet"
)

RANDOM_STATE = 42

K_VALUES = range(3, 10)

FINAL_K = 6

CLUSTER_NAMES = {
    0: "Big Ticket Shoppers",
    1: "High Frequency Generalists",
    2: "Promotion-Led Shoppers",
    3: "Category Specialists",
    4: "Omnichannel Mainstream",
    5: "Store-Led Shoppers",
}


# ---------------------------------------------------------------------
# Features used by clustering
# ---------------------------------------------------------------------

CLUSTER_FEATURES = [
    # Purchase intensity
    "orders",
    "avg_order_value",
    "units_per_order",
    "avg_margin_per_order",

    # Promotion behaviour
    "avg_discount_pct",
    "discounted_order_share",

    # Channel behaviour
    "store_share",
    "online_share",
    "click_collect_share",
    "channels_used",

    # Category breadth / concentration
    "categories_used",
    "category_concentration",
    "dominant_category_share",

    # Cadence
    "median_purchase_gap_days",
    "cadence_cv",
]


LOG_FEATURES = [
    "orders",
    "avg_order_value",
    "avg_margin_per_order",
    "median_purchase_gap_days",
]


# ---------------------------------------------------------------------
# Prepare feature matrix
# ---------------------------------------------------------------------

def prepare_features(
    customers: pd.DataFrame,
):

    matrix = customers[
        CLUSTER_FEATURES
    ].copy()

    # Reduce skew while preserving relative differences.
    for column in LOG_FEATURES:
        matrix[column] = np.log1p(
            matrix[column]
        )

    # Customers with insufficient repeat history should remain
    # clusterable rather than being excluded.
    imputer = SimpleImputer(
        strategy="median"
    )

    imputed = imputer.fit_transform(
        matrix
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        imputed
    )

    return (
        scaled,
        imputer,
        scaler,
    )


# ---------------------------------------------------------------------
# Evaluate candidate K values
# ---------------------------------------------------------------------

def evaluate_clusters(
    scaled_features: np.ndarray,
) -> pd.DataFrame:

    diagnostics = []

    for k in K_VALUES:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(
            scaled_features
        )

        silhouette = silhouette_score(
            scaled_features,
            labels,
            sample_size=min(
                10000,
                len(labels),
            ),
            random_state=RANDOM_STATE,
        )

        counts = pd.Series(
            labels
        ).value_counts()

        diagnostics.append(
            {
                "k": k,
                "silhouette_score":
                    silhouette,
                "inertia":
                    model.inertia_,
                "smallest_cluster":
                    counts.min(),
                "largest_cluster":
                    counts.max(),
                "smallest_cluster_pct":
                    counts.min()
                    / len(labels),
                "largest_cluster_pct":
                    counts.max()
                    / len(labels),
            }
        )

    return pd.DataFrame(
        diagnostics
    )


# ---------------------------------------------------------------------
# Choose candidate K
# ---------------------------------------------------------------------

def choose_candidate_k(
    diagnostics: pd.DataFrame,
) -> int:

    # Avoid solutions containing tiny clusters unless they provide
    # exceptional separation.
    viable = diagnostics.loc[
        diagnostics[
            "smallest_cluster_pct"
        ] >= 0.03
    ].copy()

    if viable.empty:
        viable = diagnostics.copy()

    best_row = (
        viable
        .sort_values(
            "silhouette_score",
            ascending=False,
        )
        .iloc[0]
    )

    return int(
        best_row["k"]
    )


# ---------------------------------------------------------------------
# Fit final candidate model
# ---------------------------------------------------------------------

def fit_final_model(
    scaled_features: np.ndarray,
    k: int,
):

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=30,
    )

    labels = model.fit_predict(
        scaled_features
    )

    return model, labels


# ---------------------------------------------------------------------
# PCA coordinates for visualisation
# ---------------------------------------------------------------------

def build_pca_coordinates(
    scaled_features: np.ndarray,
):

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    coordinates = pca.fit_transform(
        scaled_features
    )

    return (
        coordinates,
        pca.explained_variance_ratio_,
    )


# ---------------------------------------------------------------------
# Build output
# ---------------------------------------------------------------------

def build_cluster_output(
    customers: pd.DataFrame,
    labels: np.ndarray,
    pca_coordinates: np.ndarray,
) -> pd.DataFrame:

    output = customers.copy()

    output["cluster"] = labels

    output["cluster_name"] = (
        output["cluster"]
        .map(CLUSTER_NAMES)
    )

    output["pca_1"] = (
        pca_coordinates[:, 0]
    )

    output["pca_2"] = (
        pca_coordinates[:, 1]
    )

    return output


# ---------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------

def run_qa(
    clustered: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_k: int,
    explained_variance: np.ndarray,
) -> None:

    print("\nCUSTOMER CLUSTERING QA")
    print("=" * 80)

    print(
        f"Golden customers clustered: "
        f"{len(clustered):,}"
    )

    print(
        f"Unique golden customer IDs: "
        f"{clustered['golden_customer_id'].nunique():,}"
    )

    print(
        f"Duplicate golden customer IDs: "
        f"{clustered['golden_customer_id'].duplicated().sum():,}"
    )

    print("\nK-MEANS DIAGNOSTICS")
    print("-" * 80)

    display = diagnostics.copy()

    display[
        "silhouette_score"
    ] = (
        display[
            "silhouette_score"
        ]
        .round(3)
    )

    display["inertia"] = (
        display["inertia"]
        .round(0)
    )

    display[
        "smallest_cluster_pct"
    ] = (
        display[
            "smallest_cluster_pct"
        ]
        .mul(100)
        .round(1)
    )

    display[
        "largest_cluster_pct"
    ] = (
        display[
            "largest_cluster_pct"
        ]
        .mul(100)
        .round(1)
    )

    print(
        display.to_string(
            index=False
        )
    )

    print(
        f"\nCandidate K selected: "
        f"{selected_k}"
    )

    print(
        "\nCluster sizes:"
    )

    cluster_sizes = (
        clustered["cluster"]
        .value_counts()
        .sort_index()
    )

    print(cluster_sizes)

    print(
        "\nCluster share (%):"
    )

    print(
        (
            cluster_sizes
            / len(clustered)
            * 100
        )
        .round(1)
    )

    print(
        "\nPCA explained variance:"
    )

    print(
        f"PC1: "
        f"{explained_variance[0]:.1%}"
    )

    print(
        f"PC2: "
        f"{explained_variance[1]:.1%}"
    )

    print(
        f"Combined: "
        f"{explained_variance.sum():.1%}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CLUSTERED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading clustering features..."
    )

    customers = pd.read_parquet(
        INPUT_FILE
    )

    if "golden_customer_id" not in customers.columns:
        raise ValueError(
            "Clustering features must contain "
            "golden_customer_id."
        )

    if customers["golden_customer_id"].duplicated().any():
        raise ValueError(
            "Clustering features must contain "
            "one row per golden_customer_id."
        )

    missing_features = (
        set(CLUSTER_FEATURES)
        - set(customers.columns)
    )

    if missing_features:
        raise ValueError(
            "Clustering features are missing "
            f"required model inputs: "
            f"{sorted(missing_features)}"
        )

    print(
        "Preparing feature matrix..."
    )

    (
        scaled_features,
        _,
        _,
    ) = prepare_features(
        customers
    )

    print(
        "Evaluating candidate cluster counts..."
    )

    diagnostics = evaluate_clusters(
        scaled_features
    )

    # selected_k = choose_candidate_k(
    #     diagnostics
    # )

    # K=6 selected based on statistical diagnostics,
    # cluster stability and commercial interpretability.
    selected_k = FINAL_K

    print(
        "Fitting candidate clustering model..."
    )

    _, labels = fit_final_model(
        scaled_features,
        selected_k,
    )

    (
        pca_coordinates,
        explained_variance,
    ) = build_pca_coordinates(
        scaled_features
    )

    clustered = build_cluster_output(
        customers,
        labels,
        pca_coordinates,
    )

    diagnostics.to_csv(
        DIAGNOSTICS_FILE,
        index=False,
    )

    clustered.to_parquet(
        CLUSTERED_FILE,
        index=False,
    )

    run_qa(
        clustered,
        diagnostics,
        selected_k,
        explained_variance,
    )

    print("\nFiles created:")
    print(DIAGNOSTICS_FILE)
    print(CLUSTERED_FILE)


if __name__ == "__main__":
    main()