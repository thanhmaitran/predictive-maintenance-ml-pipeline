# Phase 16 (Milestone 8): reproducible environment for the full
# predictive-maintenance pipeline, running on top of the official
# Airflow image so `docker compose up` gives a working Airflow install
# AND our pipeline code/dependencies in one image.
#
# Version pinned to match requirements.txt's apache-airflow==2.9.3.
FROM apache/airflow:2.9.3-python3.11

WORKDIR /opt/airflow

# Install our pipeline's Python dependencies on top of Airflow's own
# (deliberately excludes apache-airflow itself — see requirements-docker.txt).
COPY --chown=airflow:root requirements-docker.txt /opt/airflow/requirements-docker.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements-docker.txt

# Bake the project into the image so `docker build` alone produces a
# runnable artifact. docker-compose.yml additionally mounts these same
# paths as volumes for local development (edit on host, no rebuild needed).
COPY --chown=airflow:root src/ /opt/airflow/src/
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root config/ /opt/airflow/config/
COPY --chown=airflow:root tests/ /opt/airflow/tests/

# Data/model/report directories are populated at runtime by the pipeline
# (or mounted from the host) — just ensure they exist with correct ownership.
USER root
RUN mkdir -p /opt/airflow/data/raw /opt/airflow/data/interim /opt/airflow/data/validated /opt/airflow/data/processed \
    /opt/airflow/models /opt/airflow/reports/figures /opt/airflow/reports/metrics \
    && chown -R airflow:root /opt/airflow/data /opt/airflow/models /opt/airflow/reports
USER airflow

ENV PYTHONPATH="/opt/airflow"
