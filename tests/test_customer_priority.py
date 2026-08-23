import pandas as pd

from src.decision_engine.customer_priority import (
    build_lapse_score,
    build_priority_score,
)


def test_priority_score_weights():

    df = pd.DataFrame(
        {
            "value_score": [100],
            "lapse_risk_score": [80],
            "momentum_risk_score": [60],
        }
    )

    result = build_priority_score(df)

    expected = (
        0.40 * 100
        + 0.35 * 80
        + 0.25 * 60
    )

    assert (
        result.loc[
            0,
            "priority_score",
        ]
        == round(expected, 1)
    )


def test_low_confidence_customer_gets_zero_lapse_score():

    df = pd.DataFrame(
        {
            "adjusted_lapse_ratio": [3.0],
            "cadence_confidence": ["Low"],
        }
    )

    result = build_lapse_score(df)

    assert (
        result.loc[
            0,
            "lapse_risk_score",
        ]
        == 0
    )