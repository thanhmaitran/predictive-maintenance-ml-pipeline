"""
Cleaning / transformation stage (Phase 6).

Runs only after data has PASSED the Data Quality Gate.

Responsibilities:
    - Normalise column names (e.g. strip units into consistent snake_case).
    - Derive `product_quality` (L/M/H) from Type.
    - Drop identifiers not usable as ML features (UDI, Product ID) while
      keeping them available for traceability/reporting if needed.
    - Handle duplicates according to the agreed strategy (dataset is
      expected to have none if validation passed).

Does NOT do feature engineering (temperature_difference, angular_speed,
estimated_power, wear_torque_index) — that lives in
src/features/feature_engineering.py.
"""

import pandas as pd

RENAME_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# Columns kept in the cleaned dataset for traceability/reporting, but never
# passed into the ML feature matrix (that filtering happens explicitly at
# train/test-split time using config.yaml's `columns.numeric` /
# `columns.categorical` / `columns.engineered` lists).
ID_COLUMNS = ["udi", "product_id"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a validated raw DataFrame into an analysis-ready form.

    Args:
        df: Validated DataFrame (post Data Quality Gate).

    Returns:
        Cleaned DataFrame with normalised (snake_case) column names.
        Identifier columns (udi, product_id) are RETAINED for traceability
        but must be excluded from the ML feature matrix downstream — see
        ID_COLUMNS and config.yaml's `columns.id_columns`.
    """
    cleaned = df.copy()

    missing_expected = [c for c in RENAME_MAP if c not in cleaned.columns]
    if missing_expected:
        raise ValueError(f"Cannot clean: missing expected raw columns {missing_expected}")

    cleaned = cleaned.rename(columns=RENAME_MAP)

    # Sanity check: `product_id`'s leading letter should match `type`
    # (e.g. "L47181" -> type "L"). This is a soft consistency check, not a
    # hard requirement — it is only used to log a warning, since `type` is
    # already the authoritative categorical column.
    mismatched = cleaned[cleaned["product_id"].str[0] != cleaned["type"]]
    if len(mismatched) > 0:
        print(
            f"Warning: {len(mismatched)} row(s) where product_id's leading "
            "letter does not match `type`."
        )

    # Duplicates should already be 0 post-validation, but drop defensively.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped = before - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} duplicate row(s) during cleaning.")

    # Reorder columns: identifiers first (for traceability), then features,
    # then failure-mode columns, then target last.
    ordered_cols = (
        ID_COLUMNS
        + ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
        + ["twf", "hdf", "pwf", "osf", "rnf"]
        + ["machine_failure"]
    )
    cleaned = cleaned[ordered_cols]

    return cleaned


if __name__ == "__main__":
    df = pd.read_csv("data/validated/ai4i2020_validated.csv")
    cleaned = clean(df)
    cleaned.to_csv("data/interim/cleaned_data.csv", index=False)
    print(f"Cleaned data written: {len(cleaned)} rows, {len(cleaned.columns)} columns.")
