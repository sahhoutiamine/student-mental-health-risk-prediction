
import json
import joblib
import pandas as pd
import streamlit as st

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Student Mental Health", layout="wide")
st.title("Student Mental Health Risk Analysis")
st.warning("Educational prototype only. Not a medical diagnostic or screening tool.")

# Load saved results
model = joblib.load(ROOT / "models/best_model.joblib")
metrics = json.loads((ROOT / "models/final_metrics.json").read_text())
results = pd.read_csv(ROOT / "models/baseline_results.csv", index_col=0)
cm = pd.read_csv(ROOT / "models/confusion_matrix.csv", index_col=0)

tab1, tab2 = st.tabs(["Model Performance", "Prediction"])

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

with tab2:
    st.header("Student Information")

    X_train = pd.read_csv(ROOT / "data/processed/train/X_train.csv")

    values = {}

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        for i, column in enumerate(X_train.columns):
            with [col1, col2][i % 2]:
                if pd.api.types.is_numeric_dtype(X_train[column]):
                    values[column] = st.number_input(
                        column,
                        value=float(X_train[column].median())
                    )
                else:
                    options = sorted(X_train[column].dropna().unique())
                    values[column] = st.selectbox(column, options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        student = pd.DataFrame([values])
        prediction = model.predict(student)[0]

        st.success(f"Predicted category: {prediction}")
        st.caption(
            "This prediction is an ML demonstration, "
            "not an individual mental health assessment."
        )
