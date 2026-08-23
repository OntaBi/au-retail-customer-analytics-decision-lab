import pandas as pd

from src.customer.customer_features import (
    add_active_flag,
    add_lapse_features,
)


def test_active_customer_window():

    df = pd.DataFrame(
        {
            "days_since_last_purchase": [
                100,
                365,
                366,
            ]
        }
    )

    result = add_active_flag(
        df,
        active_window_days=365,
    )

    assert result["active_customer"].tolist() == [
        True,
        True,
        False,
    ]


def test_adjusted_lapse_ratio_uses_minimum_expected_return():

    df = pd.DataFrame(
        {
            "days_since_last_purchase": [4],
            "median_purchase_gap_days": [2],
        }
    )

    result = add_lapse_features(df)

    assert result.loc[
        0,
        "expected_return_days",
    ] == 7

    assert round(
        result.loc[
            0,
            "adjusted_lapse_ratio",
        ],
        3,
    ) == round(
        4 / 7,
        3,
    )