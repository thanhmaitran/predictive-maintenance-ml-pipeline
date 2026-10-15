"""
Inference on a single new observation (Phase 13).

Loads the selected model_pipeline.joblib (preprocessing + estimator
bundled together) and runs a raw observation through:

    raw input -> feature engineering -> preprocessing -> model -> probability

Example output:
    Failure probability: 0.73
    Prediction: MACHINE FAILURE
"""

import math
from typing import Any, Dict

import joblib
import pandas as pd

from src.models.train import FEATURE_COLUMNS

RPM_TO_RAD_PER_SEC = 2 * math.pi / 60

# Raw input keys expected from a caller (mirrors the original AI4I column
# names, since that's the natural interface for a new observation coming
# straight from a sensor/PLC feed rather than the internal snake_case
# schema used inside the pipeline).
RAW_TO_INTERNAL = {
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
}

DEFAULT_MODEL_PATH = "models/model_pipeline.joblib"
DECISION_THRESHOLD = 0.5


def _engineer_single_observation(observation: Dict[str, Any]) -> pd.DataFrame:
    """Rename raw keys to internal schema and derive the 4 engineered features."""
    missing = [k for k in RAW_TO_INTERNAL if k not in observation]
    if missing:
        raise ValueError(f"Observation is missing required fields: {missing}")

    row = {internal: observation[raw] for raw, internal in RAW_TO_INTERNAL.items()}
    df = pd.DataFrame([row])

    df["temperature_difference"] = df["process_temperature"] - df["air_temperature"]
    angular_speed = df["rotational_speed"] * RPM_TO_RAD_PER_SEC
    df["angular_speed"] = angular_speed
    df["estimated_power"] = df["torque"] * angular_speed
    df["wear_torque_index"] = df["tool_wear"] * df["torque"]

    return df[FEATURE_COLUMNS]


def predict(observation: Dict[str, Any], model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Predict machine-failure probability for a single new observation.

    Args:
        observation: Dict with keys Type, Air temperature [K],
            Process temperature [K], Rotational speed [rpm],
            Torque [Nm], Tool wear [min].
        model_path: Path to the fitted model_pipeline.joblib.

    Returns:
        Dict with keys "failure_probability" (float) and
        "prediction" ("MACHINE FAILURE" or "NO FAILURE").
    """
    pipeline = joblib.load(model_path)
    X = _engineer_single_observation(observation)

    proba = float(pipeline.predict_proba(X)[0, 1])
    prediction = "MACHINE FAILURE" if proba >= DECISION_THRESHOLD else "NO FAILURE"

    return {"failure_probability": proba, "prediction": prediction}


if __name__ == "__main__":
    example = {
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
    }
    result = predict(example)
    print(f"Failure probability: {result['failure_probability']:.2f}")
    print(f"Prediction: {result['prediction']}")
