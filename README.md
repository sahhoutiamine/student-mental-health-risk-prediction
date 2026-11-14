# Student Mental Health Risk Prediction

**An end-to-end, educational machine learning project exploring factors associated with student well-being and classifying a simulated mental-health risk label into three categories: Low, Medium, and High.**

The repository combines exploratory data analysis (EDA), data quality checks, cleaning, reproducible train/test splitting, scikit-learn preprocessing pipelines, classification benchmarks, hyperparameter tuning, model evaluation, a Streamlit demonstration, and a Docker-based environment for Streamlit, JupyterLab, and Apache Airflow.

> [!IMPORTANT]
> **Educational demonstration only.** This project is **not** a medical diagnostic or screening system. Its classifications must not be used for clinical assessment, disciplinary action, admission, student ranking, or automated allocation of student services. The dataset is described as **simulated**; model performance on these records does not establish clinical validity or real-world reliability.

## Contents

- [Project overview](#project-overview)
- [Dataset](#dataset)
- [Exploratory analysis and data quality](#exploratory-analysis-and-data-quality)
- [Data preparation](#data-preparation)
- [Machine learning methodology](#machine-learning-methodology)
- [Results](#results)
- [Streamlit application](#streamlit-application)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Getting started](#getting-started)
- [Reproducing the analysis](#reproducing-the-analysis)
- [Airflow and Docker architecture](#airflow-and-docker-architecture)
- [Limitations and future improvements](#limitations-and-future-improvements)
- [Ethics and responsible use](#ethics-and-responsible-use)
- [Project context](#project-context)

## Project overview

University students may face academic demands, financial pressure, limited support, irregular sleep, and other challenges. This project investigates **associations** between such self-reported characteristics and a simulated risk label, then demonstrates a multiclass supervised classification workflow.

**Project goals**

1. Explore relationships between lifestyle, academic, and psychosocial variables and the provided target label.
2. Audit missingness, duplicates, outliers, and inconsistent records.
3. Prepare features while keeping model-fitted transformations inside a training pipeline.
4. Compare Logistic Regression, Random Forest, and Support Vector Classification (SVC).
5. Tune model hyperparameters and evaluate the selected classifier.
6. Save the trained preprocessing-and-model pipeline and present metrics and sample predictions through Streamlit.
7. Provide a containerized environment and Airflow service configuration for orchestration.

**Workflow at a glance**

```text
Simulated student CSV
        |
        v
EDA and quality audit (01_eda)
        |
        v
Consistency correction + feature selection + 80/20 stratified split (02_cleaning)
        |
        +--> data/processed/train/
        +--> data/processed/test/
        |
        v
Train-only fit: imputation + scaling + categorical encoding
        |
        v
Logistic Regression | Random Forest | SVC
        |
        v
GridSearchCV (5-fold CV, macro F1) -> selected pipeline
        |
        v
Test evaluation + error analysis -> joblib / JSON / CSV artifacts
        |
        v
Streamlit: model comparison + input form + predicted class
```

## Dataset

The supplied dataset description identifies a **simulated student survey** with **2,000 observations** and **17 columns**: a student identifier, 15 other input attributes, and the target `Mental_Health_Risk`. For modeling, `Student_ID`, `Age`, and `Gender` are dropped, leaving **13 input features**.

| Property | Value |
|---|---|
| Dataset file expected by notebooks | `data/raw/student_mental_health.csv` |
| Observations | 2,000 |
| Original columns | 17 |
| Model input features | 13 |
| Target | `Mental_Health_Risk` |
| Prediction type | Three-class classification |
| Target labels | `Low`, `Medium`, `High` |

### Model input features

| Category | Features |
|---|---|
| Academic | `Year_of_Study`, `Study_Hours_Per_Week`, `GPA`, `Attendance_Rate` |
| Lifestyle | `Sleep_Hours`, `Social_Media_Hours`, `Screen_Time_Hours`, `Physical_Activity`, `Diet_Quality` |
| Psychosocial and personal circumstances | `Financial_Stress`, `Family_Support`, `Part_Time_Job`, `Previous_Mental_Health_History` |

The original dataset also includes `Student_ID`, `Age`, and `Gender`, which are **excluded from the trained model** in `02_cleaning.ipynb`. `Previous_Mental_Health_History` is a sensitive feature and requires particular care in any future research.

### Target distribution

| Class | Records | Share |
|---|---:|---:|
| High | 680 | 34% |
| Medium | 660 | 33% |
| Low | 660 | 33% |
| **Total** | **2,000** | **100%** |

The target is already approximately balanced. **No SMOTE or other resampling step is used in the supplied training notebook.**

## Exploratory analysis and data quality

The notebook `01_eda.ipynb` explores:

- dataset shape, data types, sample records, summary statistics, and observed ranges;
- missing values, duplicate rows, and missing target values;
- logical consistency between daily screen time and social-media time;
- IQR-based outlier counts and boxplots;
- target balance and class-based distributions of categorical/numerical features;
- Pearson correlation heatmap and mutual-information ranking;
- sleep-hours relationships with risk class and GPA.

### Audit findings

| Finding | Result |
|---|---:|
| Duplicate rows | 0 |
| Missing target labels | 0 |
| Rows where `Screen_Time_Hours < Social_Media_Hours` | 37 |
| Missing `Financial_Stress` | 91 (4.55%) |
| Missing `Diet_Quality` | 69 (3.45%) |
| Missing `Sleep_Hours` | 62 (3.10%) |
| Missing `Social_Media_Hours` | 58 (2.90%) |
| Missing `Family_Support` | 54 (2.70%) |
| Missing `Attendance_Rate` | 47 (2.35%) |
| Missing `GPA` | 38 (1.90%) |

The EDA notebook also **flags**, but does not automatically delete, IQR outliers. Example counts include 20 for `Social_Media_Hours`, 12 for `Attendance_Rate`, and 9 for `GPA`.

In an exploratory mutual-information calculation, `Sleep_Hours` had the highest reported score (approximately **0.161**), followed by `GPA` (**0.075**) and `Financial_Stress` (**0.050**). These are **dataset-specific associations, not evidence of causation or clinical importance**. The exploratory computation included variables that the final model later excludes, and was not a held-out feature-selection study.

## Data preparation

The notebook `02_cleaning.ipynb` performs the following operations:

1. Loads `data/raw/student_mental_health.csv`.
2. Fixes 37 logical inconsistencies by setting `Screen_Time_Hours` equal to `Social_Media_Hours` whenever the former is smaller.
3. Separates the target `Mental_Health_Risk` from the inputs.
4. Excludes `Student_ID`, `Age`, and `Gender` from the model input.
5. Uses `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
6. Writes separate feature and target CSVs to `data/processed/train/` and `data/processed/test/`.

| Split | Records | Input columns |
|---|---:|---:|
| Training | 1,600 (80%) | 13 |
| Test | 400 (20%) | 13 |

The training target contains **544 High, 528 Medium, and 528 Low** examples.

### Preprocessing fitted during model training

`03_model_training.ipynb` constructs a `ColumnTransformer` as part of each scikit-learn `Pipeline`:

| Feature type | Missing-value handling | Transformation |
|---|---|---|
| Numeric | `SimpleImputer(strategy="median")` | `StandardScaler` |
| Ordinal (`Physical_Activity`, `Diet_Quality`) | Most frequent category | `OrdinalEncoder` with explicit category order |
| Remaining categorical | Most frequent category | `OneHotEncoder(handle_unknown="ignore")` |

Ordinal orders are `Low < Moderate < High` for physical activity and `Poor < Average < Good` for diet quality. Unknown ordinal values are encoded as `-1`.

Because the imputer, encoders, and scaler are inside the estimator pipeline, they are fitted on the training data during model fitting and inside the cross-validation folds during hyperparameter search.

**Methodology note:** The screen-time consistency correction occurs before the train/test split but uses a row-wise rule rather than statistics learned from the dataset. IQR outliers are assessed during EDA; a learned outlier-treatment transformer is **not** included in the final model pipeline.

## Machine learning methodology

### Models compared

1. **Logistic Regression** — linear multiclass classification baseline (`max_iter=1000`).
2. **Random Forest** — nonlinear ensemble of decision trees.
3. **SVC** — support-vector classifier with probability estimates enabled.

All three are trained on the same 1,600-row training split using the same feature-preprocessing design.

### Hyperparameter optimization

`GridSearchCV` searches model-specific hyperparameter grids with:

- **5-fold cross-validation** on the training split;
- **`f1_macro`** as the selection metric;
- parallel execution through `n_jobs=-1`;
- class weighting set to `balanced` in the displayed tuning grids.

The recorded searches evaluate 3 Logistic Regression configurations, 16 Random Forest configurations, and 6 SVC configurations.

| Model | Best cross-validation macro F1 |
|---|---:|
| Logistic Regression | **0.752** |
| SVC | 0.747 |
| Random Forest | 0.693 |

The notebook selects **Logistic Regression** based on the highest cross-validation macro F1 score.

## Results

### Baseline model comparison — test split

These values were printed by the supplied training notebook:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Multiclass ROC-AUC (OvR) |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.765** | **0.765** | **0.764** | **0.764** | **0.913** |
| Random Forest | 0.705 | 0.703 | 0.704 | 0.703 | 0.855 |
| SVC | 0.730 | 0.733 | 0.730 | 0.731 | 0.890 |

### Selected model — recorded final evaluation

| Metric | Value |
|---|---:|
| Selected classifier | **Logistic Regression** |
| Accuracy | **0.765** |
| Macro precision | **0.765** |
| Macro recall | **0.764** |
| Macro F1-score | **0.764** |
| Multiclass ROC-AUC (one-vs-rest) | **0.913** |
| Errors on 400 test examples | **94** |

**Per-class performance:**

| Risk class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| High | 0.83 | 0.84 | 0.83 | 136 |
| Low | 0.82 | 0.80 | 0.81 | 132 |
| Medium | 0.65 | 0.65 | 0.65 | 132 |

The `Medium` class is notably harder to classify. The recorded error analysis lists 25 `Low → Medium`, 23 `Medium → High`, 23 `Medium → Low`, 22 `High → Medium`, and 1 `Low → High` misclassification.

> [!CAUTION]
> **Evaluation caveat:** The notebook compares baseline models on the test split *before* completing tuning and final reporting. Although the tuned model is selected using training-set cross-validation, inspecting the test metrics earlier weakens the claim that the test set was untouched during development. For a stricter evaluation, reserve the test set until all model and tuning decisions are final and use cross-validation (or a separate validation split) for model comparison.

### Saved outputs

The training notebook writes:

```text
models/
├── best_model.joblib       # preprocessing + fitted classifier
├── baseline_results.csv    # baseline metric comparison
├── final_metrics.json      # selected model evaluation metrics
└── confusion_matrix.csv    # final confusion matrix
```

The `.joblib` artifact contains the **full preprocessing-and-classifier pipeline**. Load only artifacts from trusted sources, because joblib/pickle deserialization can execute arbitrary code.

## Streamlit application

The provided `app/streamlit_app.py` exposes two tabs:

**Model Performance**
- Metric cards loaded from `models/final_metrics.json`.
- Table comparing baseline models from `models/baseline_results.csv`.
- F1-score bar chart.
- Confusion matrix shown as a data table.

**Prediction**
- Dynamically generated form based on the saved training feature columns.
- Numeric inputs and category selectors.
- A **Predict** button calling `best_model.predict()` on one DataFrame row.
- An explicit message stating the prediction is an ML demonstration, **not** an individual mental-health assessment.

**Actual scope:** the supplied app does **not** implement a CSV upload feature, probability breakdown, a plotted confusion-matrix heatmap, or clinical guidance. Those are potential future improvements, not current features. The form uses the training columns, so `data/processed/train/X_train.csv` is required alongside the model artifacts.

## Project structure

The following structure reflects the provided repository layout and configured paths. Only some files were supplied for this review; directories such as `src/`, `tests/`, and the Airflow DAG itself were **not verified** from the attachments.

```text
student-mental-health-risk-prediction/
├── airflow/
│   ├── dags/                 # DAG location configured in Docker Compose
│   ├── logs/                 # Airflow execution logs
│   └── plugins/              # optional Airflow plugins
├── app/
│   └── streamlit_app.py      # Streamlit dashboard and prediction form
├── data/
│   ├── raw/
│   │   └── student_mental_health.csv
│   ├── interim/
│   └── processed/
│       ├── train/
│       │   ├── X_train.csv
│       │   └── y_train.csv
│       └── test/
│           ├── X_test.csv
│           └── y_test.csv
├── models/                  # generated model + metrics artifacts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   └── 03_model_training.ipynb
├── reports/                 # optional exported charts and reports
├── src/                     # intended reusable Python modules (not reviewed)
├── tests/                   # intended automated tests (not reviewed)
├── Dockerfile               # multi-stage app / airflow images
├── docker-compose.yml       # PostgreSQL, Airflow, Streamlit, optional Jupyter
├── requirements.txt         # required by Dockerfile; not supplied for review
├── .env.example             # expected by setup; not supplied for review
└── README.md
```

For the notebook files shared in this conversation, the EDA and training files had parenthesized copy suffixes (e.g. `01_eda(3).ipynb`); the tree uses the intended clean repository names. Adjust filenames if yours differ.

## Technology stack

| Layer | Technology | Verified use |
|---|---|---|
| Language | Python 3.11 | App Docker base image; Airflow Python 3.11 image |
| Data wrangling | pandas, NumPy | Data exploration and cleaning |
| Exploration | Matplotlib, Seaborn | Plots, target distributions, correlations, boxplots |
| Feature analysis | scikit-learn mutual information | Exploratory feature ranking |
| Machine learning | scikit-learn | Pipelines, models, metrics, GridSearchCV |
| Serialization | joblib | Saved fitted pipeline |
| Web application | Streamlit | Interactive model demonstration |
| Workflow infrastructure | Apache Airflow 2.9.3 | Dockerfile and Compose services |
| Metadata database | PostgreSQL 16 Alpine | Airflow metadata DB in Compose |
| Containers | Docker / Docker Compose | Multi-service local setup |
| Notebook environment | JupyterLab | Optional Compose `dev` profile |

**Not demonstrated in the supplied model notebook:** XGBoost as a classifier, SHAP explanations, SMOTE, or an `imblearn` pipeline. The Dockerfile comment references XGBoost's system dependency, but that does not demonstrate its use in the training experiments.

## Getting started

### Prerequisites

- Docker Engine and Docker Compose v2 for the containerized setup, **or** Python with the required dependencies for local notebook/Streamlit execution.
- The dataset at `data/raw/student_mental_health.csv`.
- A matching `requirements.txt` (the Dockerfile expects it) and any `.env.example` file used by the repository.

### Option A — Docker Compose

```bash
git clone <YOUR-REPOSITORY-URL>
cd student-mental-health-risk-prediction

# Create a local environment file from the repository template:
cp .env.example .env

# Put your dataset into data/raw/student_mental_health.csv

# Initialize Airflow metadata database and administrator:
docker compose up airflow-init

# Start PostgreSQL, Airflow and Streamlit:
docker compose up -d --build

# Optional notebook environment:
docker compose --profile dev up -d jupyter
```

On Linux, the Compose configuration reads `AIRFLOW_UID` from `.env` (default `50000`); configuring it to match the current user can help avoid file-ownership problems. Review/change default passwords and the Airflow webserver secret before exposing any service to a network.

| Service | Local URL | Description |
|---|---|---|
| Streamlit | http://localhost:8501 | Dashboard and sample prediction form |
| Airflow UI | http://localhost:8080 | DAG monitoring and logs |
| JupyterLab | http://localhost:8888 | Notebook environment under `dev` profile |

```bash
docker compose ps
docker compose logs -f airflow-scheduler
docker compose logs -f streamlit
docker compose down

# WARNING: also deletes PostgreSQL's persisted volume:
docker compose down -v
```

**Important:** the Streamlit app loads saved model/metrics files at startup. On a fresh clone, first generate the processed data and model artifacts as described below. Otherwise the app will fail when these files do not exist.

### Option B — Run locally

```bash
python -m venv .venv

# Linux/macOS:
source .venv/bin/activate

# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# After generating the artifacts:
streamlit run app/streamlit_app.py
```

Airflow is configured for Docker Compose in the provided setup; a standalone local Airflow installation requires separate setup and version-compatible dependencies.

## Reproducing the analysis

The notebooks use **relative paths** such as `../data/raw/...`; open them with the working directory set to `notebooks/`.

1. Place the simulated CSV at `data/raw/student_mental_health.csv`.
2. Run **`notebooks/01_eda.ipynb`** to inspect data quality, distributions, correlations, and feature associations.
3. Run **`notebooks/02_cleaning.ipynb`** to apply the screen-time correction and create the stratified train/test files.
4. Run **`notebooks/03_model_training.ipynb`** to fit the three classifiers, tune them with GridSearchCV, evaluate, and export model artifacts.
5. Start Streamlit and review the **Model Performance** and **Prediction** tabs.

These are notebook-driven steps confirmed by the supplied files. No standalone CLI training command or verified complete Airflow DAG implementation was included in the reviewed attachments.

## Airflow and Docker architecture

`docker-compose.yml` defines these services:

```text
PostgreSQL (metadata)
      |
      +--> airflow-init (one-time migration/admin setup)
      +--> airflow-webserver (UI, port 8080)
      +--> airflow-scheduler (DAG scheduling)

Shared project directories (data/, models/, reports/)
      |
      +--> Streamlit (port 8501)
      +--> JupyterLab (port 8888, optional dev profile)
```

- Airflow uses `LocalExecutor` with PostgreSQL metadata storage.
- The Airflow containers mount `airflow/dags/`, `airflow/logs/`, and `airflow/plugins/`.
- Streamlit mounts `app/`, `src/`, `data/`, `models/`, and `reports/`.
- JupyterLab is available only when the optional `dev` Compose profile is enabled.
- Airflow task execution status is available in its web UI **when DAGs are present**.

**Verification boundary:** the Airflow DAG Python file, its schedule, task dependencies, retries, and its successful execution were not included in the supplied materials. Therefore they are **not claimed as implemented** here. A future DAG could cover data audit → preparation → training → evaluation → artifact publication, with explicit logging and retry behavior.

## Limitations and future improvements

- **Synthetic data:** results on simulated records cannot establish effectiveness with real students.
- **Not a clinical label:** the target is a dataset category, not an independently validated mental-health diagnosis.
- **Evaluation design:** use training-only CV for all model comparison and reserve the holdout test set strictly for the last evaluation.
- **Outliers:** IQR diagnostics were performed, but a formal train-fitted outlier transformation was not implemented.
- **Class balancing:** not currently necessary from the observed 34/33/33 class shares; SMOTE was not implemented.
- **Moderate classification:** the `Medium` class has a substantially lower F1 score (0.65) than `High` and `Low`.
- **Fairness and robustness:** assess subgroup performance and sensitivity to missing data before drawing conclusions.
- **Explainability:** add carefully evaluated explanations and distinguish associations from causal effects.
- **Application UX:** add artifact-missing error handling, input validation, charts, and optionally a clearly labeled batch demo.
- **Automation:** verify and document the actual Airflow DAG code and execution schedule.
- **Reproducibility:** pin dependency versions, document the dataset's origin/permissions, and include tests for the pipeline.

## Ethics and responsible use

Mental-health-related information can be highly sensitive even when used for educational purposes.

- Avoid collecting or publishing identifiable real student data.
- Do not commit raw sensitive records, private keys, passwords, or `.env` values.
- Report false negatives as well as average scores; accuracy alone is not sufficient.
- Review performance across relevant groups, without treating those analyses as permission for automated profiling.
- Protect access to model artifacts and application data.
- Treat every prediction as an **illustrative output**, never as a recommendation about an individual student.
- Any real-world study would require an appropriate legal basis, privacy/security measures, ethics review, qualified professional oversight, and independent validation.

## Project context

This is an individual educational Data Science project prepared during **9–14 November 2026**, with a mock jury presentation scheduled for **16 November 2026**. Its purpose is to demonstrate a complete applied machine-learning workflow—from EDA and structured preprocessing through supervised model comparison, evaluation, and interactive presentation.

## License and data availability

No license file or dataset redistribution permission was provided for verification. **Do not assume the dataset or source code is licensed for reuse.** Add a suitable `LICENSE` file if you intend to grant permissions for others to use your code, and cite the original data source/terms if you publish the repository.

---

**Reminder:** the model estimates labels in a simulated dataset. It must **never** be used to diagnose, screen, rank, penalize, or make decisions about real students.
