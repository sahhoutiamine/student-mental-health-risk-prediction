# Student Mental Health Risk Prediction

End-to-end Data Science project that analyses the factors associated with student well-being and builds a supervised classification prototype to estimate a mental health risk level. The project covers the full chain: data audit, cleaning, exploratory analysis, feature engineering, preprocessing, model training, hyperparameter optimisation, evaluation, orchestration with Airflow and visualisation with Streamlit.

## Important disclaimer

This project is strictly educational. The model is a pedagogical prototype and must never be presented or used as:

- a diagnostic tool,
- a clinical screening instrument,
- a basis for disciplinary decisions,
- a tool for selecting or ranking students.

Any real-world use would require clinical validation, ethical review, data protection compliance (GDPR) and supervision by qualified professionals.

## Business objectives

- Risk detection: identify students showing a higher likelihood of psychological difficulties.
- Prevention: anticipate support needs.
- Factor analysis: understand which factors are associated with psychological risk.
- Guidance: orient students towards suitable resources.
- Optimisation: help prioritise student support actions.

## Project scope

1. Loading and exploratory data analysis: structure, missing values, duplicates, inconsistencies, outliers, descriptive statistics, target distribution and class imbalance, relationships with the target, correlations.
2. Data preparation: 80/20 stratified train/test split, imputation, outlier handling (IQR or Z-score), categorical encoding, class balancing (SMOTE if needed), scaling, all grouped in a leak-free pipeline. Each cleaning decision is documented and justified.
3. Model training and comparison: at least three classifiers trained on the same training set and compared with accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC.
4. Optimisation and evaluation: GridSearchCV or RandomizedSearchCV on the most promising models, selection of the model best suited to the business context, final evaluation on the independent test set, error analysis, limitations, and export of the complete pipeline (preprocessing and model) with joblib.
5. Orchestration and application: Airflow DAG (data preparation, preprocessing, training, evaluation) with scheduling and monitoring, and a Streamlit app to consult metrics and test the model on new data.

## Repository structure

```
student-mental-health-risk-prediction/
|-- airflow/
|   |-- dags/                 Airflow DAG definitions
|   |-- logs/                 Airflow logs (ignored by git)
|   `-- plugins/              Custom Airflow plugins (optional)
|-- app/                      Streamlit application (streamlit_app.py)
|-- data/
|   |-- raw/                  Original dataset, never modified
|   |-- interim/              Intermediate cleaned data
|   `-- processed/            Final datasets used for modelling
|-- docs/                     Additional documentation and cleaning decisions
|-- models/                   Saved pipeline (.joblib), ignored by git
|-- notebooks/                EDA and experiments
|-- reports/
|   |-- figures/              Generated plots
|   `-- metrics/              Metrics and evaluation outputs (JSON, CSV)
|-- src/
|   |-- data/                 Loading, auditing and cleaning
|   |-- features/             Feature engineering and preprocessing pipeline
|   |-- models/               Training, tuning, evaluation, prediction
|   `-- utils/                Configuration, logging, helpers
|-- tests/                    Unit tests (pytest)
|-- .dockerignore
|-- .env.example              Template for environment variables
|-- .gitignore
|-- docker-compose.yml        Local stack (Airflow, PostgreSQL, Streamlit, Jupyter)
|-- Dockerfile                Multi-stage image (app and airflow targets)
|-- README.md
`-- requirements.txt
```

## Technology stack

- Language: Python 3.11
- Data: pandas, NumPy, SciPy
- Machine learning: scikit-learn, imbalanced-learn (SMOTE), XGBoost, SHAP
- Visualisation: Matplotlib, Seaborn, Plotly
- Orchestration: Apache Airflow 2.9 (LocalExecutor, PostgreSQL)
- Application: Streamlit
- Packaging: joblib, Docker, Docker Compose
- Quality: pytest, ruff, black

## Getting started

### Option 1: Docker (recommended)

Prerequisites: Docker and Docker Compose v2.

```bash
git clone <repository-url>
cd student-mental-health-risk-prediction

cp .env.example .env
# Linux only: set AIRFLOW_UID to the output of `id -u` in .env

# Place the dataset in data/raw/

docker compose up airflow-init        # initialise Airflow database and admin user
docker compose up -d --build          # start Airflow, PostgreSQL and Streamlit

# Optional: JupyterLab for notebooks
docker compose --profile dev up -d jupyter
```

Services:

| Service | URL | Notes |
|---|---|---|
| Airflow UI | http://localhost:8080 | Credentials set in `.env` (default admin / admin) |
| Streamlit app | http://localhost:8501 | Requires a trained pipeline in `models/` |
| JupyterLab | http://localhost:8888 | Only with the `dev` profile |

Useful commands:

```bash
docker compose logs -f airflow-scheduler
docker compose ps
docker compose down                   # stop services
docker compose down -v                # stop and delete the Airflow database volume
```

### Option 2: Local environment

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

For a local Airflow installation, install Apache Airflow separately with the official constraints file matching your Python version.

## Machine learning pipeline

All transformations are fitted on the training set only and wrapped in a single `imblearn.pipeline.Pipeline` to prevent data leakage:

1. Stratified train/test split (80/20).
2. Missing values: median imputation for numerical variables, most frequent value for categorical variables (justified per variable).
3. Outliers: IQR or Z-score treatment, applied on the training set only.
4. Encoding: one-hot or ordinal encoding depending on the variable type.
5. Class balancing: SMOTE applied inside cross-validation folds, never on the test set.
6. Scaling: standardisation of numerical variables.
7. Classifier: at least three models compared (for example Logistic Regression, Random Forest, XGBoost), then tuned with GridSearchCV or RandomizedSearchCV.

The final pipeline (preprocessing and model) is saved in `models/` as a `.joblib` file and reused by the Streamlit application.

## Airflow orchestration

The DAG in `airflow/dags/` chains the following tasks:

```
load_and_audit_data -> clean_data -> build_features -> train_models -> evaluate_model -> save_pipeline
```

- Scheduled execution (for example weekly) with `catchup=False`.
- Retries with delay and failure callbacks; optional email alerts configured through `.env`.
- Task status and logs available in the Airflow UI.

## Streamlit application

- Model performance: accuracy, precision, recall, F1-score, ROC-AUC.
- Confusion matrix and other evaluation charts.
- Prediction on new data: manual form input or CSV upload, with input validation and clear error messages.
- A visible disclaimer stating the educational nature of the project.

## Evaluation criteria

- Model quality: prediction performance on the test set, hyperparameter tuning, robustness and generalisation.
- Documentation and reproducibility: clear description of each step and each cleaning decision.
- Code efficiency: readability, structure, computational performance, error handling and input validation.

## Planning

- Individual project, 5 days: 09/11/2026 to 14/11/2026 at 23:59.
- Mock jury on 16/11/2026 (45 minutes): demonstration (10 min), code walkthrough (10 min), use-case scenarios (20 min), code review and technical questions (5 min).

## Ethical considerations

- Mental health data is sensitive: anonymise or pseudonymise all records and never commit raw data to the repository.
- Report recall and false negatives explicitly, since missing an at-risk student has a different cost than a false alarm.
- Check performance across subgroups (gender, age, study level) to detect potential bias.
- Predictions are indicators for discussion, never decisions.

## License

Educational use only. Add a license file before any public distribution.
