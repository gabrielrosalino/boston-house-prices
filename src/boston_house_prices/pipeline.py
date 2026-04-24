from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boston_house_prices.data import FEATURE_COLUMNS


def build_baseline_pipeline() -> Pipeline:
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor, FEATURE_COLUMNS)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def build_production_pipeline() -> Pipeline:
    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessor, FEATURE_COLUMNS)],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def build_tuned_pipeline() -> GridSearchCV:
    pipeline = build_production_pipeline()
    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 2, 4],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
    )


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
