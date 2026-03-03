# Phase 3 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

In Phase 2 we built two models with default settings. Logistic Regression won with ~80% Recall. Random Forest only managed ~49%.

Phase 3 asked: **"Can we do better?"**

We did three things:
1. Read which features the model actually relied on (Feature Importance)
2. Found the optimal settings for Random Forest (Hyperparameter Tuning)
3. Saved the best model to a file for use in Phase 4

Result: Tuned Random Forest (~82% Recall) beat Logistic Regression (~80% Recall).

---

## Step by Step — What Happened

### Step 1 — Feature Importance

Random Forest records how many times each column was used to split data across all 100 trees. The more a column is used, the more "important" it is.

**Top features found:**
1. TotalCharges — strongest signal (combines monthly charges + tenure in one number)
2. tenure — how long the customer has been with the company
3. MonthlyCharges — how much they pay per month
4. Contract type columns — month-to-month showed high importance

**EDA vs Model comparison:**
- EDA prediction: Contract type and MonthlyCharges would be #1
- Model result: TotalCharges was #1

Why? TotalCharges = MonthlyCharges × tenure. It captures BOTH signals in a single column. Not wrong — just more consolidated than expected.

---

### Step 2 — Hyperparameter Tuning (GridSearchCV)

**What are hyperparameters?**
Settings you choose before training. The algorithm doesn't learn them — you set them.

| Hyperparameter | What it controls | Values tried |
|---|---|---|
| n_estimators | Number of trees | 100, 200, 300 |
| max_depth | How deep each tree grows | 5, 10, None |
| min_samples_split | Min customers to make a split | 2, 10 |

**How GridSearchCV works:**
- Tries all combinations: 3 × 3 × 2 = 18 combinations
- For each combination, runs 5-fold cross-validation (trains 5 times on different data slices)
- Total: 18 × 5 = 90 models trained
- Picks the combination with the highest Recall

**Best settings found:**
```
{'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 200}
```

**Why max_depth=5 won:**
Shallow trees avoid overfitting. A deep tree memorises training data perfectly but fails on new customers — like a student who memorised the textbook word for word but can't answer a rephrased question.

---

### Step 3 — Final Comparison

| Model | Recall |
|-------|--------|
| Random Forest (Phase 2, default) | ~49% |
| Logistic Regression (Phase 2) | ~80% |
| **Random Forest (Phase 3, tuned)** | **~82%** |

**Key lesson:** Default settings can be deeply misleading. A model that looks "worse" with defaults can outperform after tuning. Never ship a model without at least basic hyperparameter search.

---

### Step 4 — Saved Model to File

The winning model was saved as `models/best_model.pkl`.

**What .pkl means:** Python's "pickle" format — serialises any Python object to disk.

**Why save it:**
- The model disappears when the notebook closes
- Phase 4 loads this file to serve predictions via an API
- Production servers load from file on startup — no retraining needed

Load it back with:
```python
import pickle
model = pickle.load(open("models/best_model.pkl", "rb"))
prediction = model.predict(new_customer_data)
```

---

## Key Concepts — Memorise These

### Feature Importance
- Tells you which columns the model relied on most
- Calculated from how many times each feature was used to split data across all trees
- Most important ≠ most correlated — a feature could be important because it corrects mistakes from other features

### Overfitting
- When a model learns the training data too well — including noise and randomness
- Performs great on training data, poorly on new data
- Prevented by: limiting tree depth, requiring minimum samples to split, cross-validation

### Cross-Validation (cv=5)
- Splits training data into 5 equal chunks
- Trains on 4 chunks, tests on 1 — rotates 5 times
- Each combination gets 5 scores — average is used
- Prevents lucky results from a single train/test split

### Overfitting vs Underfitting
| | Training Score | Test Score | Problem |
|--|--|--|--|
| Overfitting | Very high | Much lower | Model memorised training data |
| Underfitting | Low | Low | Model is too simple |
| Good fit | High | Close to training | Model generalised well |

---

## Interview Questions — Phase 3

### Feature Importance

**Q: What is feature importance and how is it calculated in Random Forest?**
> Feature importance measures how much each column contributed to the model's predictions. In Random Forest, it's calculated by tracking how many times each feature was used to split data across all trees, weighted by how much each split reduced impurity (uncertainty). Higher importance = more splits used that feature.

**Q: What were your most important features and did they match your EDA findings?**
> TotalCharges was #1, followed by tenure and MonthlyCharges. In EDA I predicted Contract type would be strongest based on the 40% churn rate gap between month-to-month and two-year contracts. TotalCharges topped the model because it equals MonthlyCharges × tenure — capturing both signals in one column. Contract type still appeared in the top 5, so the EDA finding held — it just wasn't the single top feature.

**Q: Can you remove low-importance features to improve the model?**
> Yes — this is called feature selection. Removing features with near-zero importance reduces noise in the model and speeds up training. However, you should validate that removing them doesn't hurt your key metric. Sometimes a low-importance feature contributes small but consistent signal.

---

### Hyperparameter Tuning

**Q: What is the difference between a parameter and a hyperparameter?**
> Parameters are learned by the model from training data — for example, the weight assigned to each feature in Logistic Regression. Hyperparameters are settings you choose before training — like how many trees in a Random Forest or how deep they can grow. The model never learns hyperparameters; you must set or search for them.

**Q: Explain GridSearchCV in simple terms.**
> GridSearchCV systematically tries every combination of hyperparameter values you specify. For each combination, it uses cross-validation to evaluate performance, then returns the combination with the best score. It's like trying every combination of oven temperature and baking time to find the best cake — exhaustive but guaranteed to find the best option within your grid.

**Q: What is overfitting and how did you prevent it?**
> Overfitting is when a model learns training data too precisely — including noise — and fails on new data. I prevented it by: (1) limiting max_depth to 5 so trees ask fewer questions and learn general patterns rather than memorising, (2) using cross-validation in GridSearchCV so the best settings are validated on unseen data, not just measured once.

**Q: Why did you use Recall as the scoring metric in GridSearchCV instead of accuracy?**
> Because our goal is to catch as many churners as possible — missing a churner costs more than a false alarm. Using accuracy would optimise for the easy majority class (stayed customers). Using recall forces GridSearchCV to find settings that maximise catching actual churners, which aligns with the business goal.

**Q: What is cross-validation and why is it important during tuning?**
> Cross-validation splits training data into k folds. For each hyperparameter combination, the model trains on k-1 folds and evaluates on the remaining fold — rotating k times. The average score across all folds is used. This matters during tuning because without it, you could pick settings that happened to work well on one particular train/test split by chance. Cross-validation gives a reliable, averaged estimate.

**Q: What hyperparameters did you tune and what were the best values?**
> I tuned n_estimators (100, 200, 300), max_depth (5, 10, None), and min_samples_split (2, 10). Best values: max_depth=5, n_estimators=200, min_samples_split=2. max_depth=5 was the most impactful — shallow trees generalised better to unseen customers than deep trees which overfitted.

---

### Model Saving

**Q: How do you save and load a trained model in Python?**
> Using pickle: `pickle.dump(model, open("model.pkl", "wb"))` to save, and `pickle.load(open("model.pkl", "rb"))` to load. The model is serialised — converted to bytes — and written to disk. Loading restores it to the exact same state without retraining.

**Q: Why save the model instead of just retraining when needed?**
> Retraining takes time and compute resources — for large datasets this could take hours. Saving means you train once and deploy anywhere. It also ensures consistency — the exact same model is used in production as was evaluated. Retraining could produce slightly different results due to randomness.

**Q: What is the difference between saving a model and saving a pipeline?**
> A model only saves the trained algorithm. A pipeline saves the full sequence: preprocessing steps (like scaling or encoding) + the model. Saving a pipeline is safer because it guarantees the same transformations applied during training are also applied to new data at prediction time. We'll cover this in Phase 4.

---

## Combined Interview Story — All 3 Phases

If asked "walk me through your ML project end to end":

> "I worked on a customer churn prediction project for a telecom dataset of 7,032 customers.
>
> In Phase 1, I explored the data and found it was imbalanced — 74% stayed, 26% churned. I identified key patterns: month-to-month contract customers churned at 43% vs 3% for two-year contracts. I then cleaned the data — fixed a TotalCharges column stored as text, removed 11 blank rows, and converted all categorical columns to numbers using one-hot encoding.
>
> In Phase 2, I trained Logistic Regression and Random Forest baselines. Logistic Regression achieved 80% Recall. Random Forest with default settings only managed 49% — showing that default settings can be misleading.
>
> In Phase 3, I used Random Forest feature importance to confirm TotalCharges, tenure, and MonthlyCharges were the top predictors. I then ran GridSearchCV across 18 hyperparameter combinations with 5-fold cross-validation scored on Recall. The best settings — max_depth=5, n_estimators=200 — improved Random Forest Recall to 82%, beating the Logistic Regression baseline. I saved the winning model to a .pkl file for deployment in Phase 4."

---

## What's Next — Phase 4

- Build a REST API using FastAPI — any application can send customer data and get a churn prediction back
- Containerise with Docker — so it runs the same on any machine or cloud server
- This is where the DevOps/cloud skills you already have start to connect with the ML work
