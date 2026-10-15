"""
Domain-informed feature engineering (Phase 7).

Derives four engineering features from the raw sensor columns:

    temperature_difference = process_temperature - air_temperature
    angular_speed           = rotational_speed * 2*pi / 60        (rad/s)
    estimated_power         = torque * angular_speed              (W)
    wear_torque_index       = tool_wear * torque

These are motivated directly by the AI4I 2020 failure-mode definitions
(HDF relates to temperature difference; PWF relates to power; OSF relates
to tool-wear x torque), giving the pipeline a domain-knowledge story
rather than arbitrary feature bloat.
"""

import math

import pandas as pd

RPM_TO_RAD_PER_SEC = 2 * math.pi / 60


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four domain-derived features to a cleaned DataFrame.

    Args:
        df: Cleaned DataFrame containing air/process temperature,
            rotational speed, torque and tool wear columns (snake_case,
            as produced by src.preprocessing.preprocess.clean).

    Returns:
        DataFrame with temperature_difference, angular_speed,
        estimated_power and wear_torque_index columns appended.
    """
    required = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot engineer features: missing columns {missing}")

    out = df.copy()

    out["temperature_difference"] = out["process_temperature"] - out["air_temperature"]

    angular_speed = out["rotational_speed"] * RPM_TO_RAD_PER_SEC
    out["angular_speed"] = angular_speed

    out["estimated_power"] = out["torque"] * angular_speed

    out["wear_torque_index"] = out["tool_wear"] * out["torque"]

    return out


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cleaned_data.csv")
    featured = engineer_features(df)
    featured.to_csv("data/processed/features.csv", index=False)
    new_cols = ["temperature_difference", "angular_speed", "estimated_power", "wear_torque_index"]
    print(f"Feature engineering done: {len(featured)} rows, {len(featured.columns)} columns.")
    print(featured[new_cols].describe())
