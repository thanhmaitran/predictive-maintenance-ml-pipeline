"""
Airflow DAG orchestrating the end-to-end predictive-maintenance pipeline
(Phase 15 / Milestone 7).

    ingest_data -> validate_data -> clean_data -> feature_engineering
    -> split_data -> train_model -> evaluate_model -> save_model

Important: this DAG should only be built AFTER the underlying Python
pipeline (src/) runs correctly standalone (Milestones 2-6, done). Airflow
here orchestrates existing, already-working scripts (see
dags/pipeline_tasks.py) — it does not contain pipeline logic itself.

`validate_data` is a genuine Data Quality Gate: pipeline_tasks.py raises
DataQualityGateError on failure, which Airflow surfaces as a normal task
failure — all downstream tasks (clean_data onward, including train_model)
are then automatically NOT run, since Airflow only proceeds along a
dependency edge after the upstream task succeeds.

To run for real: `pip install apache-airflow==2.9.3` (pinned in
requirements.txt), place/symlink this repo's `dags/` folder as your
AIRFLOW_HOME/dags, `airflow standalone`, then trigger
`predictive_maintenance_dag` from the UI or `airflow dags trigger`.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline_tasks import TASK_SEQUENCE

default_args = {
    "owner": "portfolio",
    "retries": 0,
}

with DAG(
    dag_id="predictive_maintenance_dag",
    description="AI4I 2020 predictive maintenance: ingest -> validate -> clean -> feature-engineer -> split -> train -> evaluate -> save",
    default_args=default_args,
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["predictive-maintenance", "portfolio"],
) as dag:

    tasks = {
        task_id: PythonOperator(task_id=task_id, python_callable=callable_)
        for task_id, callable_ in TASK_SEQUENCE
    }

    (
        tasks["ingest_data"]
        >> tasks["validate_data"]
        >> tasks["clean_data"]
        >> tasks["feature_engineering"]
        >> tasks["split_data"]
        >> tasks["train_model"]
        >> tasks["evaluate_model"]
        >> tasks["save_model"]
    )
