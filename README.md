# Customer Churn Prediction — MLOps Pipeline

An end-to-end MLOps project that predicts telecom customer churn using machine learning, served via a REST API, with CI/CD and experiment tracking.

![CI](https://github.com/ashishm641/customer-churn-mlops-aws/actions/workflows/ci.yml/badge.svg)

---

## Problem Statement

A telecom company wants to predict which customers are likely to cancel their service (**churn**) so they can intervene early (offer discounts, better support, etc.).

- **Dataset**: IBM Telco Customer Churn — 7,043 customers, 21 features
- **Target**: Churn (Yes/No)
- **Key challenge**: Imbalanced data (only 26.5% churned)

---

## Project Architecture

```
Data (CSV) → EDA → Data Cleaning → Model Training → MLflow Tracking
                                         ↓
                                   best_model.pkl
                                         ↓
                                   FastAPI REST API (/predict)
                                         ↓
                                   GitHub Actions CI (auto-test on every push)
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 73% | 49% | 80% | 61% |
| Random Forest (default) | 79% | 63% | 49% | 55% |
| **Random Forest (tuned)** | **72%** | **49%** | **81%** | **61%** |

**Best model**: Random Forest (tuned) with **81% Recall** — catches 81 out of 100 actual churners.

**Why Recall?** Missing a churner (false negative) is costlier than a false alarm. A missed churner = lost revenue.

---

## Key Findings from EDA

- Month-to-month contracts have **43% churn** vs 3% for two-year contracts
- New customers (short tenure) churn the most
- Customers paying higher monthly charges churn more
- No tech support = 25% higher churn rate
- Top predictors: TotalCharges, tenure, MonthlyCharges, Contract type

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| ML Models | Scikit-learn (Logistic Regression, Random Forest) |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |
| Containerization | Docker (Dockerfile ready) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## Project Structure

```
├── api/
│   └── main.py                  # FastAPI app with /predict and /health endpoints
├── data/
│   ├── raw/telco_churn.csv      # Original dataset (7,043 rows)
│   └── processed/telco_churn_clean.csv  # Cleaned dataset (7,032 rows, 31 features)
├── docs/                        # Phase summaries & interview prep (Phases 1-6)
├── models/
│   └── best_model.pkl           # Trained Random Forest model
├── notebooks/
│   ├── 01_look_at_the_data.ipynb
│   ├── 02_find_patterns.ipynb
│   ├── 03_clean_the_data.ipynb
│   ├── 04_build_first_model.ipynb
│   └── 05_improve_the_model.ipynb
├── scripts/
│   ├── download_data.py         # Download IBM Telco dataset
│   └── train_with_mlflow.py     # Train models with MLflow tracking
├── tests/
│   └── test_api.py              # 5 automated API tests
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI pipeline
├── Dockerfile                   # Container recipe for deployment
├── requirements.txt             # Full project dependencies
└── requirements-api.txt         # Lightweight API-only dependencies
```

---

## How to Run

### 1. Clone and Setup
```bash
git clone https://github.com/ashishm641/customer-churn-mlops-aws.git
cd customer-churn-mlops-aws
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- Swagger docs: `http://127.0.0.1:8000/docs`

### 3. Run Tests
```bash
pytest tests/test_api.py -v
```

### 4. Train Models with MLflow Tracking
```bash
python scripts/train_with_mlflow.py
mlflow ui
```

---

## API Example

```
POST /predict

Request:
{
  "tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
  "Contract_Two_year": 0, "InternetService_Fiber_optic": 1, ...
}

Response:
{
  "churn_prediction": 1,
  "churn_probability": 0.839,
  "message": "This customer is likely to churn."
}
```

---

## Phases

| Phase | Description | Key Deliverable |
|-------|------------|----------------|
| 1 | Data exploration & cleaning | Clean dataset (7,032 rows, 31 features) |
| 2 | Model building (LR vs RF) | Baseline models with evaluation |
| 3 | Hyperparameter tuning | Best model saved (81% Recall) |
| 4 | REST API (FastAPI) | `/predict` endpoint |
| 5 | CI/CD (GitHub Actions) | Automated testing on every push |
| 6 | Experiment tracking (MLflow) | Dashboard with all model runs |