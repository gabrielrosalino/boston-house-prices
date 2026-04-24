from __future__ import annotations

import pandas as pd

from boston_house_prices.data import FEATURE_COLUMNS
from boston_house_prices.pipeline import build_baseline_pipeline, evaluate_regression


def test_baseline_pipeline_can_fit_and_predict() -> None:
    features = pd.DataFrame([{column: float(index + 1) for index, column in enumerate(FEATURE_COLUMNS)} for _ in range(8)])
    target = pd.Series([20.0, 21.0, 19.0, 22.0, 25.0, 24.0, 23.0, 26.0])

    model = build_baseline_pipeline()
    model.fit(features, target)
    predictions = model.predict(features)

    assert len(predictions) == len(target)


def test_evaluate_regression_returns_expected_keys() -> None:
    metrics = evaluate_regression([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert all(isinstance(value, float) for value in metrics.values())
