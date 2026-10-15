"""Tests for src/models/predict.py (Phase 13)."""

import joblib
import pandas as pd

from src.models.predict import _engineer_single_observation, predict
from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN, _build_pipelines

EXAMPLE_OBSERVATION = {
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
}


def _train_dummy_pipeline(tmp_path):
    """Train a tiny pipeline and save it so predict() has something to load."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 100
    rpm = rng.normal(1500, 150, n)
    torque = rng.normal(40, 10, n)
    air_temp = rng.normal(300, 2, n)
    process_temp = air_temp + rng.normal(10, 1, n)
    tool_wear = rng.uniform(0, 250, n)
    angular_speed = rpm * 2 * np.pi / 60
    df = pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], size=n),
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
        "temperature_difference": process_temp - air_temp,
        "angular_speed": angular_speed,
        "estimated_power": torque * angular_speed,
        "wear_torque_index": tool_wear * torque,
        TARGET_COLUMN: rng.integers(0, 2, n),
    })
    pipeline = _build_pipelines()["random_forest"]
    pipeline.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])
    model_path = tmp_path / "model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_engineer_single_observation_has_all_feature_columns():
    X = _engineer_single_observation(EXAMPLE_OBSERVATION)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 1


def test_engineer_single_observation_formula_correctness():
    X = _engineer_single_observation(EXAMPLE_OBSERVATION)
    assert abs(X.loc[0, "temperature_difference"] - (308.6 - 298.1)) < 1e-9


def test_predict_returns_expected_keys(tmp_path):
    model_path = _train_dummy_pipeline(tmp_path)
    result = predict(EXAMPLE_OBSERVATION, model_path=str(model_path))
    assert set(result.keys()) == {"failure_probability", "prediction"}
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["prediction"] in {"MACHINE FAILURE", "NO FAILURE"}


def test_predict_missing_field_raises():
    incomplete = dict(EXAMPLE_OBSERVATION)
    del incomplete["Torque [Nm]"]
    try:
        _engineer_single_observation(incomplete)
        assert False, "expected ValueError for missing field"
    except ValueError:
        pass
