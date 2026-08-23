from src.modelling.customer_clustering import (
    CLUSTER_NAMES,
    FINAL_K,
)


def test_final_cluster_configuration():

    assert FINAL_K == 6

    assert len(CLUSTER_NAMES) == 6

    assert set(
        CLUSTER_NAMES.keys()
    ) == set(
        range(FINAL_K)
    )