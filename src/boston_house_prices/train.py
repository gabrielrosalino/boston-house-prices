from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from boston_house_prices.config import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from boston_house_prices.data import load_dataset, split_features_target
from boston_house_prices.pipeline import build_production_pipeline, evaluate_regression


def train_model(
    dataset_path: Path | None = None,
    model_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, float]:
    """Train the model, persist artifacts and return evaluation metrics."""
    dataframe = load_dataset(dataset_path)
    features, target = split_features_target(dataframe)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = build_production_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = evaluate_regression(y_test, predictions)
    metrics["target"] = TARGET_COLUMN
    metrics["train_rows"] = float(len(x_train))
    metrics["test_rows"] = float(len(x_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / MODEL_FILENAME)

    with (reports_dir / METRICS_FILENAME).open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics
