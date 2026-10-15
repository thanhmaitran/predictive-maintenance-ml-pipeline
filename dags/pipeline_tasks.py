"""
Pure-Python task implementations for the Airflow DAG (Phase 15).

Deliberately contains NO Airflow imports, so these functions can be
called directly from tests or a plain Python script (see
dags/_simulate_dag_run.py) without Airflow installed — the DAG file
(predictive_maintenance_dag.py) is a thin wrapper that hooks these into
PythonOperator and translates a failed Data Quality Gate into a real
Airflow task failure.

Each task reads/writes fixed file paths (matching config/config.yaml)
rather than passing DataFrames through XCom, which is the standard
Airflow pattern for anything beyond small values.
"""

import sys
from pathlib import Path

# Allow running as a script/DAG file without the project installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# NOTE: pandas / scikit-learn / src.* are imported lazily inside each task.
# Airflow imports this file every time it parses the DAG; importing the heavy
# ML stack at module level made the DagBag import exceed the default 30 s
# timeout on Docker Desktop (AirflowTaskTimeout seen in the scheduler log).

PATHS = {
    "raw": "data/raw/ai4i2020.csv",
    "interim_raw_copy": "data/interim/raw_copy.csv",
    "validated": "data/validated/ai4i2020_validated.csv",
    "quality_report": "reports/metrics/data_quality_report.json",
    "cleaned": "data/interim/cleaned_data.csv",
    "features": "data/processed/features.csv",
    "train": "data/processed/train.csv",
    "test": "data/processed/test.csv",
    "models_dir": "models",
    "reports_dir": "reports",
}


class DataQualityGateError(Exception):
    """Raised when the Data Quality Gate fails, halting the DAG."""


def task_ingest_data() -> None:
    from src.ingestion.ingest import ingest

    ingest(raw_path=PATHS["raw"], interim_path=PATHS["interim_raw_copy"])


def task_validate_data() -> None:
    import pandas as pd

    from src.validation.validate import validate

    df = pd.read_csv(PATHS["interim_raw_copy"])
    passed = validate(df, report_path=PATHS["quality_report"])
    if not passed:
        raise DataQualityGateError(
            f"Data Quality Gate FAILED — see {PATHS['quality_report']}. "
            "Halting pipeline; train_model will not run."
        )
    Path(PATHS["validated"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS["validated"], index=False)


def task_clean_data() -> None:
    import pandas as pd

    from src.preprocessing.preprocess import clean

    df = pd.read_csv(PATHS["validated"])
    cleaned = clean(df)
    Path(PATHS["cleaned"]).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PATHS["cleaned"], index=False)


def task_feature_engineering() -> None:
    import pandas as pd

    from src.features.feature_engineering import engineer_features

    df = pd.read_csv(PATHS["cleaned"])
    featured = engineer_features(df)
    Path(PATHS["features"]).parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(PATHS["features"], index=False)


def task_split_data() -> None:
    from src.features.split import split_data

    split_data(
        features_path=PATHS["features"],
        train_path=PATHS["train"],
        test_path=PATHS["test"],
        test_size=0.2,
        random_state=42,
        stratify=True,
    )


def task_train_model() -> None:
    from src.models.train import train_models

    train_models(train_path=PATHS["train"], models_dir=PATHS["models_dir"])


def task_evaluate_model() -> None:
    from src.models.evaluate import evaluate_models

    evaluate_models(
        test_path=PATHS["test"],
        models_dir=PATHS["models_dir"],
        reports_dir=PATHS["reports_dir"],
    )


def task_save_model() -> None:
    """Confirm the standardized artifact produced by evaluate_model exists.

    evaluate_models() already writes models/model_pipeline.joblib and
    models/model_metadata.json (Phase 13). This task is the DAG's explicit
    "artifact is ready" checkpoint — kept as a separate task (rather than
    folded into evaluate_model) so the DAG graph mirrors the plan's
    8-task diagram and so a future version could add e.g. model-registry
    upload here without touching evaluation logic.
    """
    pipeline_path = Path(PATHS["models_dir"]) / "model_pipeline.joblib"
    metadata_path = Path(PATHS["models_dir"]) / "model_metadata.json"
    if not pipeline_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected model artifact at {pipeline_path} and {metadata_path} "
            "— did evaluate_model run successfully?"
        )
    print(f"Model artifact ready: {pipeline_path}")
    print(f"Model metadata ready: {metadata_path}")


TASK_SEQUENCE = [
    ("ingest_data", task_ingest_data),
    ("validate_data", task_validate_data),
    ("clean_data", task_clean_data),
    ("feature_engineering", task_feature_engineering),
    ("split_data", task_split_data),
    ("train_model", task_train_model),
    ("evaluate_model", task_evaluate_model),
    ("save_model", task_save_model),
]
