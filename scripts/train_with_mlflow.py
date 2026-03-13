"""
Model Training with MLflow Experiment Tracking
================================================
This script trains the same models from Notebooks 04 and 05,
but this time LOGS everything to MLflow so we can compare runs
in a dashboard.

What gets logged for each run:
- Parameters (model settings like max_depth, n_estimators)
- Metrics (accuracy, precision, recall, f1)
- The trained model itself (so we can load it later)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
# Tell MLflow to name our experiment (like a folder for related runs)
mlflow.set_experiment("churn-prediction")

# Load the clean data (same as Notebook 04)
df = pd.read_csv("data/processed/telco_churn_clean.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Same 80/20 split with the same random seed (so results are reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} customers")
print(f"Test set:     {X_test.shape[0]} customers")
print(f"Features:     {X_train.shape[1]}")
print("=" * 50)


def evaluate_and_log(model, model_name, params):
    """
    Train a model, evaluate it, and log everything to MLflow.
    
    This is the key function — it wraps the training in an MLflow "run"
    so that parameters, metrics, and the model are all recorded.
    """
    with mlflow.start_run(run_name=model_name):
        # 1. Train the model
        model.fit(X_train, y_train)
        
        # 2. Make predictions on test set
        y_pred = model.predict(X_test)
        
        # 3. Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # 4. Log parameters (model settings)
        mlflow.log_params(params)
        
        # 5. Log metrics (model performance)
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("precision", round(precision, 4))
        mlflow.log_metric("recall", round(recall, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        
        # 6. Log the trained model (so we can load it later from MLflow)
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        # 7. Print results
        print(f"\n{model_name}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return recall


# ---------------------------------------------------------------------------
# Run 1: Logistic Regression (same as Notebook 04)
# ---------------------------------------------------------------------------
print("\n--- Run 1: Logistic Regression ---")
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_recall = evaluate_and_log(
    lr_model,
    "Logistic Regression",
    {"model_type": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000}
)

# ---------------------------------------------------------------------------
# Run 2: Random Forest with DEFAULT settings (same as Notebook 04)
# ---------------------------------------------------------------------------
print("\n--- Run 2: Random Forest (default) ---")
rf_default = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42
)
rf_default_recall = evaluate_and_log(
    rf_default,
    "Random Forest (default)",
    {"model_type": "RandomForest", "n_estimators": 100, "max_depth": "None", "class_weight": "balanced"}
)

# ---------------------------------------------------------------------------
# Run 3: Random Forest with TUNED settings (same as Notebook 05)
# ---------------------------------------------------------------------------
print("\n--- Run 3: Random Forest (tuned) ---")
rf_tuned = RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_split=2,
    class_weight="balanced", random_state=42
)
rf_tuned_recall = evaluate_and_log(
    rf_tuned,
    "Random Forest (tuned)",
    {"model_type": "RandomForest", "n_estimators": 200, "max_depth": 5,
     "min_samples_split": 2, "class_weight": "balanced"}
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Logistic Regression:     Recall = {lr_recall:.4f}")
print(f"Random Forest (default): Recall = {rf_default_recall:.4f}")
print(f"Random Forest (tuned):   Recall = {rf_tuned_recall:.4f}")
print(f"\nBest model: ", end="")
best = max(
    [("Logistic Regression", lr_recall),
     ("Random Forest (default)", rf_default_recall),
     ("Random Forest (tuned)", rf_tuned_recall)],
    key=lambda x: x[1]
)
print(f"{best[0]} with Recall = {best[1]:.4f}")
print("\nOpen MLflow dashboard with: mlflow ui")
print("Then go to: http://127.0.0.1:5000")
