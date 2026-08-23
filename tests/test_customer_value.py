import pandas as pd

from src.customer.customer_value import (
    add_commercial_value_score,
    add_value_tier,
)


def test_no_purchase_customer_has_no_value():

    df = pd.DataFrame(
        {
            "orders": [None, 5],
            "trailing_12m_margin": [0, 500],
            "trailing_12m_sales": [0, 1500],
        }
    )

    result = add_value_tier(df)

    result = add_commercial_value_score(
        result
    )

    assert (
        result.loc[
            0,
            "customer_value_tier",
        ]
        == "No Purchase"
    )

    assert (
        result.loc[
            0,
            "commercial_value_score",
        ]
        == 0
    )