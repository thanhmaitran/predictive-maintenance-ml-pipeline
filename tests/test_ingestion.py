"""Tests for src/ingestion/ingest.py (Phase 3 / Milestone 6)."""

import pandas as pd

from src.ingestion.ingest import EXPECTED_COLUMNS, ingest


def _write_valid_csv(path):
    row = {col: 0 for col in EXPECTED_COLUMNS}
    row.update({"Type": "M", "Product ID": "M14860"})
    pd.DataFrame([row]).to_csv(path, index=False)


def test_file_exists(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    try:
        ingest(str(missing_path), str(tmp_path / "interim.csv"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_dataset_readable_by_pandas(tmp_path):
    raw_path = tmp_path / "raw.csv"
    _write_valid_csv(raw_path)
    df = ingest(str(raw_path), str(tmp_path / "interim.csv"))
    assert isinstance(df, pd.DataFrame)


def test_dataset_not_empty(tmp_path):
    raw_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(raw_path, index=False)
    try:
        ingest(str(raw_path), str(tmp_path / "interim.csv"))
        assert False, "expected ValueError for empty dataset"
    except ValueError:
        pass


def test_expected_columns_present(tmp_path):
    raw_path = tmp_path / "raw_missing_col.csv"
    row = {col: 0 for col in EXPECTED_COLUMNS if col != "Torque [Nm]"}
    pd.DataFrame([row]).to_csv(raw_path, index=False)
    try:
        ingest(str(raw_path), str(tmp_path / "interim.csv"))
        assert False, "expected ValueError for missing column"
    except ValueError:
        pass


def test_interim_copy_written(tmp_path):
    raw_path = tmp_path / "raw.csv"
    interim_path = tmp_path / "interim" / "copy.csv"
    _write_valid_csv(raw_path)
    ingest(str(raw_path), str(interim_path))
    assert interim_path.exists()
