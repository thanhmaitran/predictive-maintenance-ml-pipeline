"""
Model training (Phases 9-10).

Trains two candidate models on the stratified train split:
    1. Logistic Regression (class_weight="balanced") — interpretable baseline.
    2. Random Forest (class_weight="balanced" or equivalent) — captures
       nonlinearities/interactions and yields feature importance.

Preprocessing (OneHotEncoder for `type`, scaling for numeric columns) is
wrapped together with the estimator in a single sklearn Pipeline /
ColumnTransformer so that inference-time preprocessing and training-time
preprocessing can never drift apart. Both pipelines are fit ONLY on the
train split — the test split is never seen until evaluation.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "machine_failure"

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "temperature_difference",
    "angular_speed",
    "estimated_power",
    "wear_torque_index",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

RANDOM_STATE = 42


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _build_pipelines() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_models(train_path: str, models_dir: str) -> dict:
    """Fit Logistic Regression and Random Forest pipelines.

    Args:
        train_path: Path to data/processed/train.csv.
        models_dir: Directory to write fitted pipeline artifacts to.

    Returns:
        Dict mapping model name -> fitted sklearn Pipeline.
    """
    df = pd.read_csv(train_path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Training data missing required columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    pipelines = _build_pipelines()

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    for name, pipeline in pipelines.items():
        pipeline.fit(X, y)
        joblib.dump(pipeline, models_path / f"{name}.joblib")
        print(f"{name} trained on {len(df)} rows and saved to {models_path / f'{name}.joblib'}")

    return pipelines


if __name__ == "__main__":
    train_models(
        train_path="data/processed/train.csv",
        models_dir="models",
    )
