"""
Dataset ingestion for the AI4I 2020 predictive maintenance pipeline.

Responsibilities (Phase 3):
    - Confirm the raw dataset file exists.
    - Read it with pandas.
    - Run basic sanity checks (non-empty, expected columns present).
    - Persist a canonical interim copy for downstream stages.

This module intentionally does NOT perform validation logic (see
src/validation/validate.py) or feature engineering (see
src/features/feature_engineering.py) — ingestion is only responsible for
getting the raw data into the pipeline safely.
"""

from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def ingest(raw_path: str, interim_path: str) -> pd.DataFrame:
    """Load the raw CSV, run basic checks, and write an interim copy.

    Args:
        raw_path: Path to the raw dataset CSV (e.g. data/raw/ai4i2020.csv).
        interim_path: Path to write the canonical interim copy.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If raw_path does not exist.
        ValueError: If the dataset is empty or missing expected columns.
    """
    raw_file = Path(raw_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    # utf-8-sig handles the BOM present at the start of the AI4I CSV header.
    df = pd.read_csv(raw_file, encoding="utf-8-sig")

    if df.empty:
        raise ValueError(f"Dataset at {raw_path} is empty.")

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print(f"Ingestion OK: {raw_path}")
    print(f"  Rows received:    {len(df)}")
    print(f"  Columns received: {len(df.columns)}")

    interim_file = Path(interim_path)
    interim_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(interim_file, index=False)
    print(f"  Interim copy written to: {interim_path}")

    return df


if __name__ == "__main__":
    ingest(
        raw_path="data/raw/ai4i2020.csv",
        interim_path="data/interim/raw_copy.csv",
    )
