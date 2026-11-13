from datetime import datetime
from pathlib import Path
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator

NOTEBOOKS = Path("/opt/project/notebooks")


def run_notebook(name):
    subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", name,
            "--output", f"/tmp/{name}",
            "--ExecutePreprocessor.timeout=3600"
        ],
        cwd=NOTEBOOKS,
        check=True
    )


with DAG(
    dag_id="student_mental_health_pipeline",
    start_date=datetime(2026, 11, 12),
    schedule="0 2 * * *",
    catchup=False,
    tags=["machine-learning"]
) as dag:

    cleaning = PythonOperator(
        task_id="data_cleaning",
        python_callable=run_notebook,
        op_args=["02_cleaning.ipynb"]
    )

    training = PythonOperator(
        task_id="training_and_evaluation",
        python_callable=run_notebook,
        op_args=["03_model_training.ipynb"]
    )

    cleaning >> training
