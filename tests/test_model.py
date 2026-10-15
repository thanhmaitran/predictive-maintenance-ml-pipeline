"""Tests for src/models/train.py and evaluate.py (Milestone 6)."""

import json

import numpy as np
import pandas as pd

from src.models.train import FEATURE_COLUMNS, TARGET_COLUMN, _build_pipelines


def _make_dummy_training_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    air_temp = rng.normal(300, 2, n)
    process_temp = air_temp + rng.normal(10, 1, n)
    rpm = rng.normal(1500, 150, n)
    torque = rng.normal(40, 10, n)
    tool_wear = rng.uniform(0, 250, n)
    angular_speed = rpm * 2 * np.pi / 60
    estimated_power = torque * angular_speed
    wear_torque_index = tool_wear * torque

    # Failure more likely at high estimated_power / wear_torque_index.
    score = (estimated_power - estimated_power.mean()) / estimated_power.std() \
        + (wear_torque_index - wear_torque_index.mean()) / wear_torque_index.std()
    prob = 1 / (1 + np.exp(-score))
    machine_failure = (rng.uniform(size=n) < prob * 0.3).astype(int)

    return pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], size=n),
        "air_temperature": air_temp,
        "process_temperature": process_temp,
        "rotational_speed": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
        "temperature_difference": process_temp - air_temp,
        "angular_speed": angular_speed,
        "estimated_power": estimated_power,
        "wear_torque_index": wear_torque_index,
        "machine_failure": machine_failure,
    })


def test_model_can_train():
    df = _make_dummy_training_df()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    pipelines = _build_pipelines()
    for name, pipeline in pipelines.items():
        pipeline.fit(X, y)
        assert hasattr(pipeline.named_steps["classifier"], "classes_")


def test_model_can_predict():
    df = _make_dummy_training_df()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    pipeline = _build_pipelines()["random_forest"]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(df)


def test_prediction_shape_correct():
    df = _make_dummy_training_df()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    pipeline = _build_pipelines()["logistic_regression"]
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(df), 2)


def test_probabilities_are_valid():
    """Predicted probabilities must lie in [0, 1]."""
    df = _make_dummy_training_df()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    pipeline = _build_pipelines()["random_forest"]
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluate_ignores_existing_model_pipeline_artifact(tmp_path):
    """Regression test: evaluate_models() must not re-evaluate its own
    previously-saved model_pipeline.joblib as if it were a third candidate
    model. This was a real bug caught in Phase 20 audit — a stale
    model_pipeline.joblib from a prior run got globbed alongside
    logistic_regression.joblib / random_forest.joblib, appeared in
    model_comparison.csv as a bogus "model_pipeline" row, and (due to a
    tie-break on sorted filename order) got selected into
    model_metadata.json's "model" field instead of the real algorithm name.
    """
    import shutil
    import joblib as _joblib

    from src.models.evaluate import evaluate_models

    df = _make_dummy_training_df(n=200)
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"

    pipelines = _build_pipelines()
    for name, pipeline in pipelines.items():
        pipeline.fit(X, y)
        _joblib.dump(pipeline, models_dir / f"{name}.joblib")

    # Simulate a stale model_pipeline.joblib left over from a prior
    # evaluate_models() run, BEFORE calling evaluate_models() again.
    shutil.copy(models_dir / "random_forest.joblib", models_dir / "model_pipeline.joblib")

    test_path = tmp_path / "test.csv"
    df.to_csv(test_path, index=False)

    comparison_df = evaluate_models(str(test_path), str(models_dir), str(reports_dir))

    assert "model_pipeline" not in comparison_df.index

    with open(reports_dir / "metrics" / "evaluation.json") as f:
        evaluation = json.load(f)
    assert evaluation["model_selection"]["selected_model"] in {"logistic_regression", "random_forest"}
