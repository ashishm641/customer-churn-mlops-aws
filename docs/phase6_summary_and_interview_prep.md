# Phase 6 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

In Phases 2–3, we trained models in notebooks — results were scattered across cells, hard to compare, and easy to lose.

Phase 6 asked: **"How do we track every experiment in one place?"**

We integrated **MLflow** — an experiment tracking tool that records:
- What model we trained
- What settings (parameters) we used
- What scores (metrics) we got
- The trained model itself (saved as an artifact)

All viewable in a dashboard at http://127.0.0.1:5000.

---

## What We Built

### Files Created

| File | Purpose |
|------|---------|
| `scripts/train_with_mlflow.py` | Training script that logs 3 model runs to MLflow |

### What Gets Logged Per Run

| Category | Examples |
|----------|---------|
| **Parameters** | model_type, max_depth, n_estimators, class_weight |
| **Metrics** | accuracy, precision, recall, f1_score |
| **Artifacts** | The trained model (can be loaded directly from MLflow) |
| **Metadata** | Run name, duration, timestamp, source script |

### The 3 Runs We Logged

| Run | Recall | Key Parameters |
|-----|--------|---------------|
| Logistic Regression | 0.7968 | class_weight=balanced, max_iter=1000 |
| Random Forest (default) | 0.4893 | n_estimators=100, max_depth=None |
| Random Forest (tuned) | **0.8102** | n_estimators=200, max_depth=5 |

---

## How It Works

### Before MLflow
```
Train model in notebook → print metrics → forget them → retrain → "wait, what score did I get last time?"
```

### After MLflow
```
Train model → MLflow automatically logs params + metrics + model → permanent record → compare anytime
```

### The Key Code Pattern

```python
with mlflow.start_run(run_name="Random Forest (tuned)"):
    # 1. Train
    model.fit(X_train, y_train)
    
    # 2. Log settings
    mlflow.log_params({"max_depth": 5, "n_estimators": 200})
    
    # 3. Log scores
    mlflow.log_metric("recall", 0.8102)
    
    # 4. Log the model itself
    mlflow.sklearn.log_model(model, artifact_path="model")
```

Everything inside `with mlflow.start_run()` is recorded as one experiment run.

---

## Key Concepts Explained

### What is MLflow?

An open-source platform for managing the ML lifecycle. It has 4 main components:
1. **Tracking** — Log parameters, metrics, and models (what we used)
2. **Models** — Package models for deployment
3. **Registry** — Version and stage models (staging → production)
4. **Projects** — Package code for reproducibility

We used component #1 (Tracking).

### What is an Experiment?

A named group of related runs. Our experiment is called "churn-prediction". Think of it like a folder — all runs related to this project go here.

### What is a Run?

One training attempt. Each run records what you tried and what happened. We have 3 runs — one for each model configuration.

### What are Artifacts?

Files saved with a run. In our case, the trained model. You can also log plots, CSVs, or any file. This means you can go back to any run and load the exact model from that experiment.

### Why Not Just Print Metrics?

| print() | MLflow |
|---------|--------|
| Gone when you close the notebook | Permanently saved |
| Can't compare runs side by side | Built-in comparison UI |
| No record of parameters used | Params + metrics logged together |
| Can't reload old models | Every model saved as artifact |
| "What did I try 2 weeks ago?" | Full searchable history |

---

## Interview Questions & Answers

### Q1: What is MLflow and why did you use it?
**A:** MLflow is an open-source experiment tracking platform. I used it to log every model training run — parameters, metrics, and the model itself. This gives me a complete history of what I tried, what worked, and what didn't. Without it, experiment results are scattered across notebooks and easy to lose.

### Q2: What do you log in MLflow?
**A:** Three things per run: (1) Parameters — model settings like max_depth and n_estimators, (2) Metrics — performance scores like recall, precision, accuracy, f1, and (3) Artifacts — the trained model file so I can reload it later without retraining.

### Q3: How does MLflow help in a team setting?
**A:** Everyone on the team can see every experiment in one dashboard. If a colleague asks "what did you try?", I point them to MLflow instead of digging through notebooks. It prevents duplicate work and makes it easy to compare approaches across team members.

### Q4: What is the difference between an experiment and a run in MLflow?
**A:** An experiment is a named group (like a project folder) — e.g., "churn-prediction". A run is one training attempt within that experiment. I have one experiment with 3 runs (Logistic Regression, Random Forest default, Random Forest tuned).

### Q5: How would you use MLflow in production?
**A:** Beyond tracking, I'd use the Model Registry to version models (v1, v2, v3) and stage them (staging → production). When a new model beats the current one, I'd promote it to production. The API would load the "production" model from the registry instead of a local pickle file.

### Q6: What is `mlflow.start_run()` and why use `with`?
**A:** `start_run()` begins recording a new experiment run. Using `with` ensures the run is properly closed when the block ends — even if an error occurs. Without `with`, a crashed training could leave a run in a "running" state forever.

### Q7: Can you compare models in MLflow?
**A:** Yes. Select multiple runs in the UI and click "Compare" — it shows a side-by-side table of parameters and metrics, plus charts. This makes it easy to see which settings led to the best performance.

### Q8: How is MLflow different from just saving results in a spreadsheet?
**A:** MLflow is automated — I don't manually type results. It also saves the actual model artifact with each run, tracks the source code, timestamps everything, and provides a searchable API. A spreadsheet requires manual entry and doesn't save models.

### Q9: Where does MLflow store the data?
**A:** By default, locally in an `mlruns/` folder in your project directory. It uses a SQLite database for metadata and the file system for artifacts. For teams, you'd configure a remote tracking server with a shared database (PostgreSQL) and artifact store (S3).

### Q10: If you had to pick the best model from MLflow, how would you do it?
**A:** I'd sort runs by my primary metric (recall, in this case) and pick the highest. In code: `mlflow.search_runs(order_by=["metrics.recall DESC"])`. I'd also check that other metrics (precision, f1) aren't terrible — a model with 100% recall but 1% precision is useless.

---

## What's Next?

Remaining options for the project:
- **AWS Deployment (ECR + ECS)** — Deploy the API to the cloud
- **Model Monitoring** — Detect data drift and model degradation
- **Model Registry** — Version control for models (staging → production)
