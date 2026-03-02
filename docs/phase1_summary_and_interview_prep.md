# Phase 1 — Summary & Interview Prep

---

## What We Built (Plain English Summary)

We are building a **Customer Churn Prediction** system for a telecom company.

**The business problem:**
A telecom company is losing customers. Every customer who leaves costs money — it's 5x more expensive to get a new customer than to keep an existing one. So the company wants to predict: *"Which customers are about to leave?"* — so they can call them, offer a discount, and keep them.

**What Machine Learning does here:**
We give the model historical data — 7,043 real customers, including whether they churned or not. The model finds patterns: "customers on month-to-month contracts who pay high monthly charges tend to leave." Then when a new customer comes in, the model predicts: will they leave or stay?

---

## Phase 1 — What We Did (Step by Step)

### Checkpoint 1 — Set Up the Environment
- Created a GitHub repo: `customer-churn-mlops-aws`
- Installed Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn
- Downloaded the IBM Telco Customer Churn dataset (7,043 rows, 21 columns) and saved it to `data/raw/telco_churn.csv`

**Key learning:** Never touch the raw data file. Always work on a copy.

---

### Checkpoint 2 — Looked at the Data (Notebook 01)

**What we found:**
- 7,043 customers, 21 columns
- Target column: `Churn` (Yes = customer left, No = customer stayed)
- **26.5% churned, 73.5% stayed** — this is called an **imbalanced dataset**
- `TotalCharges` had 11 hidden blank rows (stored as text with empty strings)
- Columns include: demographics (gender, age), services (internet, phone, tech support), account info (contract type, monthly charges, tenure)

**Key learning:** Always explore data before modeling. You need to understand what you have.

---

### Checkpoint 3 — Found Patterns in the Data (Notebook 02)

This is called **EDA — Exploratory Data Analysis**. We looked at 4 columns and compared churned vs stayed customers:

| Pattern | Finding |
|---------|---------|
| Contract type | Month-to-month: ~43% churn. Two-year: ~3% churn. Huge gap. |
| Tenure | New customers (low tenure) churn much more than long-time customers |
| Monthly charges | Churned customers pay ~$15-20/month more than stayed customers |
| Tech support | Customers with NO tech support churn ~25% more than those with support |

**Strongest predictor found:** Contract type — 40% difference in churn rate between month-to-month and two-year contracts.

**Key learning:** Before building any model, find patterns with your own eyes. This is what interviewers always ask about.

---

### Checkpoint 4 — Cleaned the Data (Notebook 03)

ML models only understand numbers — not text like "Yes", "No", "Male", "Month-to-month".

**What we did:**
1. **Removed `customerID`** — it's just a random ID, no predictive value
2. **Fixed `TotalCharges`** — converted from text to number, removed 11 blank rows (7,043 → 7,032 rows)
3. **Converted Yes/No columns to 1/0** — "Yes" → 1, "No" → 0 for columns like Partner, Churn, TechSupport
4. **One-hot encoded multi-choice columns** — Contract, InternetService, PaymentMethod etc. split into separate 0/1 columns
5. **Saved clean file** to `data/processed/telco_churn_clean.csv`

**Result:** 20 columns → 31 columns (one-hot encoding creates new columns)

**Key learning:** Raw data is never model-ready. Cleaning is always required and always the most time-consuming part.

---

## Key Concepts Explained Simply

### Imbalanced Dataset
74% stayed, 26% churned. If a model just says "No churn" for everyone, it's 74% accurate but completely useless — it misses every customer about to leave. This is why accuracy alone is a bad metric for imbalanced data.

### EDA (Exploratory Data Analysis)
Looking at your data before touching any model. Like a doctor reading your medical history before prescribing medicine. You find patterns, spot problems, and understand what you're working with.

### One-Hot Encoding
Converting multi-choice text columns into numbers without creating false ordering.
- Wrong: Month-to-month=1, One year=2, Two year=3 (implies Two year is "3x bigger")
- Right: Create 3 separate columns, each with 0 or 1

### Train/Test Split (coming in Phase 2)
Split data into 80% for training (model studies this) and 20% for testing (model is examined on this — it has never seen it before). This tells us how the model will perform on real new customers.

---

## Interview Questions — Phase 1

These are real questions interviewers ask. Practice answering them out loud.

---

### Business & Problem Understanding

**Q: What is customer churn and why does it matter?**
> Churn means a customer stops using a product or service. It matters because acquiring a new customer costs 5x more than retaining one. Predicting churn lets the business take action before the customer leaves.

**Q: How would you explain this ML model to a non-technical business stakeholder?**
> We look at historical data of customers who left versus stayed. The model learns patterns — like "customers on month-to-month contracts who pay high bills tend to leave." Then for every current customer, we get a probability score: how likely are they to leave next month? The business can then target the high-risk customers with offers.

**Q: What metric would you optimize for in a churn model and why?**
> Recall — because missing a churner (false negative) is more costly than falsely flagging someone who wasn't going to churn (false positive). It's cheaper to offer a discount to someone who wasn't leaving than to lose a customer entirely.

---

### Data & EDA Questions

**Q: Walk me through your EDA process.**
> First I checked the shape — 7,043 rows, 21 columns. Then I checked the target distribution — 26.5% churned, imbalanced. I then checked for missing values — found 11 hidden blank rows in TotalCharges. Then I compared key features between churned and stayed customers — found contract type was the strongest predictor with a 40% difference in churn rate.

**Q: What patterns did you find in the data?**
> Four strong patterns: (1) Month-to-month contracts had 43% churn vs 3% for two-year — strongest signal. (2) New customers (low tenure) churned much more than long-time customers. (3) Churned customers paid higher monthly charges. (4) Customers without tech support churned ~25% more than those with it.

**Q: What is an imbalanced dataset and how do you handle it?**
> An imbalanced dataset has significantly more examples of one class than another — here 74% stayed vs 26% churned. You handle it by: (1) using better metrics like recall/F1 instead of accuracy, (2) oversampling the minority class (SMOTE), or (3) setting class_weight="balanced" in the model to penalize missing the minority class more.

**Q: What's the difference between data/raw and data/processed?**
> Raw is the original source of truth — never modified. Processed is the cleaned, transformed, model-ready version. If something goes wrong, we can always reprocess from raw.

---

### Data Cleaning Questions

**Q: Why did you remove the customerID column?**
> customerID is just a unique identifier — a random string with no relationship to churn. If left in, the model would try to learn from it and find noise patterns. Always remove ID columns before training.

**Q: What was wrong with TotalCharges and how did you fix it?**
> Despite being a numeric column, it was stored as object (text) type because 11 rows had blank strings instead of a number. We used `pd.to_numeric(errors='coerce')` to convert it — blanks became NaN — then dropped those 11 rows.

**Q: What is one-hot encoding and why do we use it instead of label encoding for this case?**
> One-hot encoding creates one binary column per category. We use it instead of label encoding (1, 2, 3...) for nominal categories like Contract type because label encoding implies a false numerical order — "Two year=3" doesn't mean it's 3x bigger than "Month-to-month=1". One-hot avoids this false relationship.

**Q: Why did your column count go from 20 to 31?**
> One-hot encoding split each multi-choice column into multiple binary columns. For example, Contract (one column, 3 values) became 2 new columns (using drop_first=True to avoid multicollinearity). Doing this across 10 multi-choice columns added 11 new columns.

---

### General ML Concepts

**Q: What is the difference between training data and test data?**
> Training data is what the model learns from — like studying for an exam. Test data is hidden from the model during training and used only at the end to evaluate performance — like the actual exam. This tells us how the model will do on new, unseen customers.

**Q: What is recall and when would you use it over accuracy?**
> Recall = out of all actual churners, how many did we catch? Accuracy = overall % correct predictions. We prefer recall when the cost of missing a positive case (churner) is high. In churn prediction, missing a churner = losing a customer. In fraud detection or disease detection, the same logic applies.

---

## What's Next — Phase 2

- Split data into train/test sets
- Train first model: Logistic Regression
- Evaluate with accuracy, recall, precision, F1
- Improve with a better model: Random Forest
- Compare results and pick the winner
