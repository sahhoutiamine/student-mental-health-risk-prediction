
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path

import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

ROOT = Path("/opt/project")
DATA = ROOT / "data"
MODELS = ROOT / "models"


# Task 1: Prepare data
def prepare_data():
    df = pd.read_csv(DATA / "raw/student_mental_health.csv")

    df = df.drop_duplicates()
    df = df.dropna(subset=["Mental_Health_Risk"])

    X = df.drop(columns=[
        "Mental_Health_Risk", "Student_ID", "Age", "Gender"
    ])
    y = df["Mental_Health_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("test", X_test, y_test)
    ]:
        folder = DATA / "processed" / name
        folder.mkdir(parents=True, exist_ok=True)

        X_part.to_csv(folder / f"X_{name}.csv", index=False)
        y_part.to_csv(folder / f"y_{name}.csv", index=False)

    print("Data prepared successfully!")


# Task 2: Train and compare models
def train_model():
    folder = DATA / "processed/train"

    X = pd.read_csv(folder / "X_train.csv")
    y = pd.read_csv(folder / "y_train.csv").squeeze("columns")

    ordinal = ["Physical_Activity", "Diet_Quality"]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [
        c for c in X.select_dtypes(exclude="number").columns
        if c not in ordinal
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),

        ("ord", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[
                    ["Low", "Moderate", "High"],
                    ["Poor", "Average", "Good"]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), ordinal),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]), categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVC": SVC(probability=True, random_state=42)
    }

    scores = {}
    best_score = -1
    best_model = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        score = cross_val_score(
            pipeline, X, y, cv=3, scoring="f1_macro"
        ).mean()

        scores[name] = score
        print(f"{name}: F1 = {score:.3f}")

        if score > best_score:
            best_score = score
            best_model = pipeline

    best_model.fit(X, y)

    MODELS.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS / "best_model.joblib")

    pd.DataFrame({
        "Model": list(scores.keys()),
        "F1-score": list(scores.values())
    }).set_index("Model").to_csv(
        MODELS / "baseline_results.csv"
    )

    print("Best model trained and saved!")


# Task 3: Evaluate model
def evaluate_model():
    folder = DATA / "processed/test"

    X_test = pd.read_csv(folder / "X_test.csv")
    y_test = pd.read_csv(folder / "y_test.csv").squeeze("columns")

    model = joblib.load(MODELS / "best_model.joblib")

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    }

    if hasattr(model, "predict_proba") and len(model.classes_) > 2:
        probabilities = model.predict_proba(X_test)
        metrics["ROC-AUC"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr",
            labels=model.classes_
        )

    with open(MODELS / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    matrix = confusion_matrix(
        y_test, predictions, labels=model.classes_
    )

    pd.DataFrame(
        matrix, index=model.classes_, columns=model.classes_
    ).to_csv(MODELS / "confusion_matrix.csv")

    print("Evaluation completed!")
    print(metrics)


# Airflow DAG
with DAG(
    dag_id="student_mental_health_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["machine-learning", "students"]
) as dag:

    task1 = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data
    )

    task2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    task3 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    task1 >> task2 >> task3
