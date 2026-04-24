from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_FILENAME = "housing.csv"
MODEL_FILENAME = "house_price_pipeline.joblib"
METRICS_FILENAME = "metrics.json"

TARGET_COLUMN = "MEDV"
RANDOM_STATE = 42
TEST_SIZE = 0.2
