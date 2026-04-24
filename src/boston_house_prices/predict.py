from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from boston_house_prices.config import MODEL_FILENAME, MODELS_DIR
from boston_house_prices.data import FEATURE_COLUMNS


def load_model(model_path: Path | None = None) -> Pipeline:
    path = model_path or MODELS_DIR / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `boston-train` before predicting."
        )
    return joblib.load(path)


def predict_house_price(payload: dict[str, float], model_path: Path | None = None) -> float:
    missing_features = set(FEATURE_COLUMNS) - set(payload)
    if missing_features:
        raise ValueError(f"Missing required features: {sorted(missing_features)}")

    dataframe = pd.DataFrame([{feature: payload[feature] for feature in FEATURE_COLUMNS}])
    model = load_model(model_path)
    prediction = model.predict(dataframe)[0]
    return float(prediction)
