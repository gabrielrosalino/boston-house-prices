from __future__ import annotations

import pandas as pd

from boston_house_prices.data import FEATURE_COLUMNS, split_features_target


def test_split_features_target_returns_expected_shapes() -> None:
    dataframe = pd.DataFrame(
        [
            {**{column: 1.0 for column in FEATURE_COLUMNS}, "MEDV": 24.0},
            {**{column: 2.0 for column in FEATURE_COLUMNS}, "MEDV": 30.0},
        ]
    )

    features, target = split_features_target(dataframe)

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.tolist() == [24.0, 30.0]
