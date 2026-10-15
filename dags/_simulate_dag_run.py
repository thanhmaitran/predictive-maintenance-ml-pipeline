"""
Simulates an Airflow DAG run by executing dags/pipeline_tasks.TASK_SEQUENCE
in order, stopping immediately if any task raises — exactly how Airflow's
`>>` dependency operator behaves (a downstream task never starts if its
upstream task failed).

This exists so the DAG wiring can be verified end-to-end in environments
where Airflow itself cannot be installed (e.g. an offline sandbox). It is
NOT a replacement for actually running `airflow standalone` + triggering
the real DAG — see the module docstring in predictive_maintenance_dag.py
for how to do that.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_tasks import TASK_SEQUENCE, DataQualityGateError


def simulate_dag_run():
    print(f"{'='*60}\nSimulating predictive_maintenance_dag\n{'='*60}")
    for task_id, task_callable in TASK_SEQUENCE:
        print(f"\n--- Task: {task_id} ---")
        start = time.time()
        try:
            task_callable()
        except DataQualityGateError as e:
            print(f"\n[TASK FAILED] {task_id}: {e}")
            print("Downstream tasks will NOT run (matches Airflow's dependency behavior).")
            return False
        except Exception as e:
            print(f"\n[TASK FAILED] {task_id}: {type(e).__name__}: {e}")
            return False
        print(f"--- Task '{task_id}' succeeded in {time.time() - start:.2f}s ---")

    print(f"\n{'='*60}\nDAG run SUCCESS — all 8 tasks completed.\n{'='*60}")
    return True


if __name__ == "__main__":
    success = simulate_dag_run()
    sys.exit(0 if success else 1)
