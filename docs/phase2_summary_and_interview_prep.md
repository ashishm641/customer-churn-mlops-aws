# Phase 2 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

In Phase 1 we cleaned the data. In Phase 2 we used that clean data to build and test two ML models.

**The goal:** Train a model that can look at a customer's details and predict — will they churn next month?

---

## Step by Step — What Happened

### Step 1 — Separated X and y

We split the 31-column dataset into two parts:

- **X** = the 30 input columns (everything the model uses to think — contract type, tenure, monthly charges etc.)
- **y** = the 1 output column (Churn: 0 = stayed, 1 = left)

**Simple analogy:** X is the question paper. y is the answer key.

---

### Step 2 — Train/Test Split

We split 7,032 customers into:
- **5,625 customers → training set** (80%) — model studies these including the answers
- **1,407 customers → test set** (20%) — hidden from model until evaluation

Why hide the test set? If the model trains and tests on the same data, it just memorises the answers — like giving a student the exam in advance. We must test on data it has never seen to know if it actually learned.

We used `stratify=y` to ensure both sets have the same 26.5% churn ratio.

---

### Step 3 — Logistic Regression (Model 1)

The simplest model for yes/no predictions. Draws one line through the data separating "likely to churn" from "likely to stay" — using all 30 features at once.

Used `class_weight="balanced"` to handle imbalanced data — this tells the model to penalise missing a churner more heavily than misclassifying someone who stayed.

**Analogy:** One experienced employee making a decision by drawing a line in their head.

**Results:**
- Accuracy:  ~79%
- Precision: ~55%
- **Recall: ~80%** ← most important
- F1 Score: ~65%

---

### Step 4 — Random Forest (Model 2)

100 decision trees each look at a random subset of features and vote. Majority wins.

Usually more accurate overall — but with `class_weight="balanced"`, it still needs a strong majority to call "churn", so it misses more actual churners.

**Analogy:** A committee of 100 people voting — harder to get agreement, so some real churners slip through undetected.

**Results:**
- Accuracy:  ~79%
- Precision: ~65%
- **Recall: ~49%** ← worse than Logistic Regression
- F1 Score: ~56%

---

### Step 5 — Model Comparison & Winner

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | ~79% | ~79% |
| Precision | ~55% | ~65% |
| **Recall** | **~80%** | **~49%** |
| F1 Score | ~65% | ~56% |

**Winner: Logistic Regression** — because Recall is the most important metric for churn. Missing a churner = losing a customer = revenue loss.

---

## Key Concepts — Memorise These

### Recall vs Precision (Most Common Interview Mix-up)

| | Definition | When it matters |
|--|--|--|
| **Recall** | Of all actual churners, what % did we catch? | When missing a positive case is costly (churn, fraud, disease) |
| **Precision** | Of all customers we flagged as churners, what % actually churned? | When false alarms are costly |

**Memory trick:** Recall = "did we RECALL (remember) all the real churners?"

---

### Confusion Matrix — 4 Boxes

```
                  Predicted: Stayed    Predicted: Churned
Actual: Stayed        True Negative        False Positive
Actual: Churned       False Negative       True Positive
```

- **True Positive (TP):** Predicted churn, actually churned ✅ 
- **True Negative (TN):** Predicted stayed, actually stayed ✅ 
- **False Positive (FP):** Predicted churn, actually stayed ❌ (sent unnecessary discount — cheap mistake)
- **False Negative (FN):** Predicted stayed, actually churned ❌ (lost the customer — expensive mistake!)

**For churn: minimise False Negatives.** That's why we optimise for Recall.

---

### Why class_weight="balanced"?

Our data is imbalanced — 74% stayed, 26% churned. Without correction, the model gets lazy and predicts "No churn" for everyone (74% accuracy, completely useless).

`class_weight="balanced"` tells the model: "treat each missed churner as more important." It automatically adjusts based on the ratio of classes.

---

### Why Does Logistic Regression Have Higher Recall Than Random Forest Here?

Logistic Regression with balanced weights shifts its decision boundary aggressively toward flagging churners — catching more of them (high recall) but also flagging some non-churners by mistake (lower precision).

Random Forest needs a majority vote across 100 trees to say "churn." This caution means higher precision but more missed churners (lower recall).

Higher accuracy ≠ better model. Always choose metrics aligned to your business goal.

---

## Interview Questions — Phase 2

### Train/Test Split

**Q: What is a train/test split and why do we do it?**
> We split data into 80% for training (model learns from this) and 20% for testing (hidden until evaluation). This simulates how the model will perform on real new customers it has never seen. Without this, the model could just memorise the training data and appear accurate without actually learning.

**Q: What is stratified split and why did you use it?**
> Stratified split preserves the class ratio in both train and test sets. Our data is 26.5% churners. Without stratify, by random chance the test set might have only 15% churners — making evaluation misleading. With `stratify=y`, both sets have ~26.5% churners.

**Q: Why 80/20 and not 50/50?**
> More training data means the model learns better patterns. 80/20 is a common convention balancing enough data to learn vs enough data to evaluate. For very large datasets, 90/10 or 95/5 is also common.

---

### Model Understanding

**Q: Explain Logistic Regression in simple terms.**
> Despite the name, it's a classification model for yes/no predictions. It finds the best line (or hyperplane in multiple dimensions) that separates churners from non-churners across all features. It outputs a probability between 0 and 1, then classifies based on a threshold (usually 0.5).

**Q: Explain Random Forest in simple terms.**
> An ensemble of decision trees. Each tree is trained on a random subset of data and features, and makes its own prediction. The final prediction is a majority vote across all trees. Because many diverse trees vote, it generally outperforms a single model and handles non-linear patterns well.

**Q: Why did you choose Logistic Regression as your baseline?**
> It is fast to train, easy to interpret, has few hyperparameters to tune, and often performs surprisingly well on structured data. It is also the standard starting point — if a simple model performs well, there is no need for a complex one. Complexity should be justified by meaningful improvement.

---

### Evaluation Metrics

**Q: Why not just use accuracy to evaluate a churn model?**
> Because the dataset is imbalanced — 74% stayed. A model that predicts "no churn" for everyone gets 74% accuracy but is completely useless. We need metrics that measure performance on the minority class specifically — precision, recall, and F1.

**Q: What is recall and why is it the most important metric for churn?**
> Recall = of all customers who actually churned, what % did the model correctly identify? It is most important because a False Negative (predicting "stayed" when they actually churned) means we lose the customer with no intervention. The cost of missing a churner is much higher than the cost of offering a discount to someone who wasn't going to leave.

**Q: What is a confusion matrix?**
> A 2x2 table showing True Positives, True Negatives, False Positives, and False Negatives. It lets you see not just how many predictions were right or wrong, but what TYPE of mistakes the model is making — which is critical for business decisions.

**Q: What is the difference between False Positive and False Negative? Which is worse for churn?**
> False Positive: model said "will churn" but they stayed — we waste a discount offer. False Negative: model said "will stay" but they churned — we lose the customer entirely. For churn, False Negatives are worse because the revenue loss from losing a customer is higher than the cost of an unnecessary discount.

**Q: What is F1 score?**
> F1 is the harmonic mean of precision and recall. It gives a single number that balances both. Useful when you care about both catching true positives AND not having too many false alarms. Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall).

---

### Model Comparison

**Q: Your Random Forest had higher accuracy than Logistic Regression but lower recall — which would you deploy?**
> Logistic Regression, because the business goal is to catch as many churners as possible before they leave. Recall directly measures this. Higher accuracy with lower recall means Random Forest is better at predicting who stays — which is the easy, majority class. We care more about the hard, minority class.

**Q: When would you choose Random Forest over Logistic Regression?**
> When the data has complex non-linear relationships that a straight line can't capture. When precision matters more than recall (e.g., you have a limited budget for outreach and can only contact a few customers — you want the ones flagged to be very likely churners). Also when you need feature importance rankings.

---

## What's Next — Phase 3

- Tune the Random Forest to improve its recall (hyperparameter tuning)
- Look at which features matter most (feature importance)
- Try to close the gap between the two models
- Save the best model to a file so it can be used in production
