
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Student Mental Health", layout="wide")
st.title("Student Mental Health Risk Analysis")

# Load model and results
model = joblib.load(ROOT / "models/best_model.joblib")
metrics = json.loads((ROOT / "models/final_metrics.json").read_text())
results = pd.read_csv(ROOT / "models/baseline_results.csv", index_col=0)
cm = pd.read_csv(ROOT / "models/confusion_matrix.csv", index_col=0)

tab1, tab2 = st.tabs(["Model Performance", "Prediction"])

# ---------------- Model Performance ----------------

with tab1:
    st.header("Model Performance")

    cols = st.columns(len(metrics))

    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

    st.subheader("Model Comparison")
    st.dataframe(results.round(3))
    st.bar_chart(results["F1-score"])

    st.subheader("Confusion Matrix")
    st.dataframe(cm)

# ---------------- Prediction ----------------

with tab2:
    st.header("Student Information")
    st.write("Enter the student's information to test the model.")

    # Numerical features: (min, max, step)
    numeric_features = {
        "Age": (18, 30, 1),
        "Year_of_Study": (1, 5, 1),
        "Sleep_Hours": (3.5, 9.5, 0.5),
        "Study_Hours_Per_Week": (5.0, 55.0, 0.5),
        "GPA": (1.5, 4.0, 0.1),
        "Social_Media_Hours": (0.5, 10.0, 0.5),
        "Screen_Time_Hours": (1.0, 14.0, 0.5),
        "Financial_Stress": (1.0, 10.0, 0.5),
        "Family_Support": (1.0, 10.0, 0.5),
        "Attendance_Rate": (30.0, 100.0, 1.0)
    }

    # Categorical features
    categorical_features = {
        "Gender": ["Male", "Female", "Non-binary"],
        "Physical_Activity": ["Low", "Moderate", "High"],
        "Part_Time_Job": ["Yes", "No"],
        "Diet_Quality": ["Poor", "Average", "Good"],
        "Previous_Mental_Health_History": ["Yes", "No"]
    }

    # Use the same features as training
    X_train = pd.read_csv(ROOT / "data/processed/train/X_train.csv")
    values = {}

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        for i, column in enumerate(X_train.columns):
            with [col1, col2][i % 2]:

                if column in numeric_features:
                    min_val, max_val, step = numeric_features[column]

                    values[column] = st.number_input(
                        column.replace("_", " "),
                        min_value=min_val,
                        max_value=max_val,
                        value=min_val,
                        step=step
                    )

                elif column in categorical_features:
                    values[column] = st.selectbox(
                        column.replace("_", " "),
                        categorical_features[column]
                    )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        student = pd.DataFrame([values])
        prediction = model.predict(student)[0]

        st.success(f"Predicted category: {prediction}")
        st.caption(
            "This prediction is an ML demonstration, "
            "not an individual mental health assessment."
        )
