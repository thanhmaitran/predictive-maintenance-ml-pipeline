"""Tests for src/features/split.py (Phase 8)."""

import numpy as np
import pandas as pd

from src.features.split import split_data


def _make_dummy_features(n=200, failure_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n_failures = int(n * failure_rate)
    machine_failure = np.array([1] * n_failures + [0] * (n - n_failures))
    rng.shuffle(machine_failure)
    return pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "type": rng.choice(["L", "M", "H"], size=n),
            "torque": rng.normal(40, 10, size=n),
            "machine_failure": machine_failure,
        }
    )


def test_split_sizes(tmp_path):
    df = _make_dummy_features(n=200)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    train_df, test_df = split_data(
        str(features_path), str(tmp_path / "train.csv"), str(tmp_path / "test.csv"),
        test_size=0.2, random_state=42, stratify=True,
    )
    assert len(train_df) == 160
    assert len(test_df) == 40


def test_split_no_overlap(tmp_path):
    df = _make_dummy_features(n=200)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    train_df, test_df = split_data(
        str(features_path), str(tmp_path / "train.csv"), str(tmp_path / "test.csv"),
        test_size=0.2, random_state=42, stratify=True,
    )
    assert set(train_df["udi"]) & set(test_df["udi"]) == set()


def test_split_is_reproducible(tmp_path):
    df = _make_dummy_features(n=200)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    train1, _ = split_data(
        str(features_path), str(tmp_path / "train1.csv"), str(tmp_path / "test1.csv"),
        test_size=0.2, random_state=42, stratify=True,
    )
    train2, _ = split_data(
        str(features_path), str(tmp_path / "train2.csv"), str(tmp_path / "test2.csv"),
        test_size=0.2, random_state=42, stratify=True,
    )
    assert set(train1["udi"]) == set(train2["udi"])


def test_stratification_preserves_failure_rate(tmp_path):
    df = _make_dummy_features(n=1000, failure_rate=0.1)
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    train_df, test_df = split_data(
        str(features_path), str(tmp_path / "train.csv"), str(tmp_path / "test.csv"),
        test_size=0.2, random_state=42, stratify=True,
    )
    overall_rate = df["machine_failure"].mean()
    train_rate = train_df["machine_failure"].mean()
    test_rate = test_df["machine_failure"].mean()
    assert abs(train_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.03


def test_missing_target_column_raises(tmp_path):
    df = _make_dummy_features(n=50).drop(columns=["machine_failure"])
    features_path = tmp_path / "features.csv"
    df.to_csv(features_path, index=False)

    try:
        split_data(
            str(features_path), str(tmp_path / "train.csv"), str(tmp_path / "test.csv"),
        )
        assert False, "expected ValueError for missing target column"
    except ValueError:
        pass
