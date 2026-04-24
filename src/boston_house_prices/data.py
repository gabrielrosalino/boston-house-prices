from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from boston_house_prices.config import DATASET_FILENAME, RAW_DATA_DIR, TARGET_COLUMN

FEATURE_COLUMNS = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]


def fetch_dataset(output_dir: Path = RAW_DATA_DIR) -> Path:
    """Download the Boston Housing dataset from OpenML and persist it as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DATASET_FILENAME

    dataset = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    dataframe = dataset.frame
    dataframe.columns = [column.upper() for column in dataframe.columns]

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in dataset.")

    dataframe.to_csv(output_path, index=False)
    return output_path


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load the dataset from disk, downloading it when necessary."""
    path = dataset_path or RAW_DATA_DIR / DATASET_FILENAME

    if not path.exists():
        path = fetch_dataset(path.parent)

    dataframe = pd.read_csv(path)
    dataframe.columns = [column.upper() for column in dataframe.columns]
    return dataframe


def split_features_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into features and target."""
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return dataframe[FEATURE_COLUMNS], dataframe[TARGET_COLUMN]
