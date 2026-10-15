"""
Train/test split (Phase 8).

Runs on the feature-engineered dataset (data/processed/features.csv) and
produces stratified train/test CSVs. Stratification matters here because
the positive class (machine_failure == 1) is only ~3.39% of the data —
without stratify=y, a random split could easily under- or over-represent
failures in the test set, and evaluation metrics like recall/F1/PR-AUC
would 	become noisy or misleading.

This intentionally runs AFTER feature engineering and BEFORE any
preprocessing that "learns" from the data (e.g. StandardScaler,
OneHotEncoder). Those must be fit on the train split only, inside the
model pipeline (see src/models/train.py), to avoid leaking test-set
statistics into training.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "machine_failure"


def split_data(
    features_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split of the feature-engineered dataset.

    Args:
        features_path: Path to data/processed/features.csv.
        train_path: Output path for the training split.
        test_path: Output path for the test split.
        test_size: Fraction of rows held out for testing.
        random_state: Seed for reproducibility (project default: 42).
        stratify: Whether to stratify on the target column. Should stay
            True given the ~3.39% failure rate.

    Returns:
        (train_df, test_df) tuple.
    """
    df = pd.read_csv(features_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in {features_path}")

    stratify_col = df[TARGET_COLUMN] if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train: {len(train_df)} rows (failure rate {train_df[TARGET_COLUMN].mean()*100:.2f}%)")
    print(f"Test:  {len(test_df)} rows (failure rate {test_df[TARGET_COLUMN].mean()*100:.2f}%)")

    return train_df, test_df


if __name__ == "__main__":
    split_data(
        features_path="data/processed/features.csv",
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv",
        test_size=0.2,
        random_state=42,
        stratify=True,
    )
