# ===============================================================
# Multi-stage Dockerfile
#   target "app"     : Streamlit application and JupyterLab
#   target "airflow" : Airflow webserver, scheduler and init
# Build examples:
#   docker build --target app -t student-mh-app .
#   docker build --target airflow -t student-mh-airflow .
# ===============================================================

# ---------------------------------------------------------------
# Stage 1: application image (Streamlit / Jupyter)
# ---------------------------------------------------------------
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/project

WORKDIR /opt/project

# System dependencies (libgomp1 is required by xgboost)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Non-root user
RUN useradd --create-home --uid 1000 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p data/raw data/interim data/processed models reports/figures reports/metrics \
    && chown -R appuser:appuser /opt/project
USER appuser

EXPOSE 8501 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ---------------------------------------------------------------
# Stage 2: Airflow image
# ---------------------------------------------------------------
FROM apache/airflow:2.9.3-python3.11 AS airflow

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/project

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/project/data /opt/project/models /opt/project/reports \
    && chown -R airflow:root /opt/project

USER airflow
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /opt/project
