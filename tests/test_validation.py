"""Tests for src/validation/validate.py (Phase 5 / Milestone 6)."""

import pandas as pd

from src.validation.validate import validate

VALID_ROW = {
    "UDI": 1,
    "Product ID": "M14860",
    "Type": "M",
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0,
    "Machine failure": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0,
}


def _make_df(n=5, **overrides):
    rows = []
    for i in range(n):
        row = dict(VALID_ROW)
        row["UDI"] = i + 1
        row["Product ID"] = f"M{14860 + i}"
        rows.append(row)
    df = pd.DataFrame(rows)
    for col, value in overrides.items():
        df.loc[0, col] = value
    return df


def test_valid_dataset_passes(tmp_path):
    df = _make_df()
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is True


def test_missing_value_fails(tmp_path):
    df = _make_df()
    df.loc[0, "Torque [Nm]"] = None
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is False


def test_invalid_product_type_fails(tmp_path):
    df = _make_df(**{"Type": "X"})
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is False


def test_negative_torque_fails(tmp_path):
    df = _make_df(**{"Torque [Nm]": -10})
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is False


def test_invalid_target_value_fails(tmp_path):
    df = _make_df(**{"Machine failure": 2})
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is False


def test_duplicate_udi_fails(tmp_path):
    df = _make_df()
    df.loc[1, "UDI"] = df.loc[0, "UDI"]
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is False


def test_failure_mode_mismatch_is_warning_not_hard_fail(tmp_path):
    """A failure-mode inconsistency alone must not fail the gate (known AI4I quirk)."""
    df = _make_df()
    df.loc[0, "TWF"] = 1  # mode fired but Machine failure stays 0
    passed = validate(df, report_path=str(tmp_path / "report.json"))
    assert passed is True


def test_no_target_leakage():
    """Failure-mode columns must never end up in the ML feature matrix."""
    feature_columns = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "temperature_difference",
        "angular_speed",
        "estimated_power",
        "wear_torque_index",
    ]
    for leaky_col in ("TWF", "HDF", "PWF", "OSF", "RNF", "Failure Type"):
        assert leaky_col not in feature_columns
