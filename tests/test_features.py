"""Tests for src/features/feature_engineering.py (Phase 7 / Milestone 6)."""

import math

import pandas as pd

from src.features.feature_engineering import engineer_features

BASE_ROW = {
    "udi": 1,
    "product_id": "M14860",
    "type": "M",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 60,
    "torque": 10.0,
    "tool_wear": 5,
    "twf": 0,
    "hdf": 0,
    "pwf": 0,
    "osf": 0,
    "rnf": 0,
    "machine_failure": 0,
}


def _make_df(**overrides):
    row = dict(BASE_ROW)
    row.update(overrides)
    return pd.DataFrame([row])


def test_angular_speed_formula():
    """rpm=60 -> angular_speed = 2*pi rad/s."""
    df = _make_df(rotational_speed=60)
    out = engineer_features(df)
    assert math.isclose(out.loc[0, "angular_speed"], 2 * math.pi, rel_tol=1e-9)


def test_estimated_power_formula():
    """rpm=60, torque=10 -> power = 10 * 2*pi."""
    df = _make_df(rotational_speed=60, torque=10.0)
    out = engineer_features(df)
    expected_power = 10.0 * 2 * math.pi
    assert math.isclose(out.loc[0, "estimated_power"], expected_power, rel_tol=1e-9)


def test_temperature_difference_formula():
    df = _make_df(air_temperature=298.1, process_temperature=308.6)
    out = engineer_features(df)
    assert math.isclose(out.loc[0, "temperature_difference"], 308.6 - 298.1, rel_tol=1e-9)


def test_wear_torque_index_formula():
    df = _make_df(tool_wear=5, torque=10.0)
    out = engineer_features(df)
    assert math.isclose(out.loc[0, "wear_torque_index"], 50.0, rel_tol=1e-9)


def test_missing_required_column_raises():
    df = _make_df().drop(columns=["torque"])
    try:
        engineer_features(df)
        assert False, "expected ValueError for missing column"
    except ValueError:
        pass
