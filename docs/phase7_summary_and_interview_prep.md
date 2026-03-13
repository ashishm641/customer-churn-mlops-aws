# Phase 7 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

We built and deployed a model, but models don't stay accurate forever. Real-world data changes over time — this is called **data drift**.

Phase 7 asked: **"How do we know when our model is going stale?"**

We built a **drift detection script** that:
1. Takes the training data (what the model learned from)
2. Compares it against new production data
3. Uses statistical tests to check if distributions have shifted
4. Generates a report showing which features drifted and by how much

---

## What We Built

### Files Created

| File | Purpose |
|------|---------|
| `scripts/detect_drift.py` | Drift detection script (compares reference vs production data) |
| `reports/data_drift_report.html` | Visual HTML report showing drift results |

### What the Script Does

```
Training data (reference) ──→ Compare distributions ──→ Drift report
Production data (current)  ──┘   (KS test per feature)
```

### Results from Our Simulation

We simulated 4 realistic changes:
1. Company raised prices by 15%
2. TotalCharges shifted accordingly
3. More new customers (shorter tenure)
4. Company pushed fiber optic plans

| Feature | KS Statistic | P-Value | Status |
|---------|-------------|---------|--------|
| MonthlyCharges | 0.1651 | 0.000000 | DRIFTED |
| tenure | 0.1229 | 0.000000 | DRIFTED |
| TotalCharges | 0.0378 | 0.000085 | DRIFTED |
| InternetService_Fiber optic | 0.0866 | 0.000000 | DRIFTED |
| InternetService_No | 0.0317 | 0.001696 | DRIFTED |
| Other 25 features | — | — | Stable |

**Overall**: 5/30 features drifted (16.7%) — below the 30% threshold, so model is still usable but should be monitored.

---

## Key Concepts Explained

### What is Data Drift?

When the distribution of incoming data changes compared to what the model was trained on.

**Example**: Your model learned that customers paying $70+/month churn more. The company raises all prices by 15%. Now most customers pay $70+. The model starts predicting everyone will churn — but the real churn rate hasn't changed. The model is wrong because the data shifted.

### What is the KS Test (Kolmogorov-Smirnov)?

A statistical test that compares two distributions and answers: "Are these significantly different?"

- **KS Statistic**: How different (0 = identical, 1 = completely different)
- **P-Value**: Probability the difference is just random noise
- **If p-value < 0.05**: The difference is real → drift detected

### Types of Drift

| Type | What Changes | Example |
|------|-------------|---------|
| **Data Drift** (what we detect) | Input feature distributions | MonthlyCharges goes up |
| **Concept Drift** | Relationship between features and target | High charges used to mean churn, but now the company offers premium features worth the price |
| **Prediction Drift** | Model output distribution | Model starts predicting more churners than usual |

We detect **data drift** — the most common and easiest to catch.

### Why Not Just Check Accuracy?

You need actual outcomes (did the customer churn?) to check accuracy. That takes months — by then the damage is done. Data drift detection works **immediately** on incoming data, no labels needed.

| Check Accuracy | Check Drift |
|---|---|
| Need actual churn outcomes | Only need input features |
| Takes weeks/months | Instant |
| Tells you "model got worse" | Tells you "model MIGHT get worse" |
| Reactive (after the fact) | Proactive (early warning) |

---

## Interview Questions & Answers

### Q1: What is data drift and why does it matter?
**A:** Data drift is when the distribution of incoming data changes compared to training data. It matters because ML models assume the future looks like the past. When that assumption breaks, predictions become unreliable. Detecting drift early lets you retrain before the model causes business damage.

### Q2: How did you detect drift in your project?
**A:** I used the Kolmogorov-Smirnov (KS) test on each feature independently. It compares the cumulative distribution of training data vs production data. If the p-value is below 0.05, that feature has drifted significantly. I then check the overall drift share — if more than 30% of features drift, the model needs retraining.

### Q3: What is the KS test and why did you choose it?
**A:** The KS test measures the maximum distance between two cumulative distribution functions. I chose it because: (1) it's non-parametric — works for any distribution shape, (2) it handles both continuous and discrete features, (3) it's simple to interpret — one statistic and one p-value per feature.

### Q4: What would you do if drift is detected?
**A:** Three steps: (1) Investigate which features drifted and why — is it a real business change or a data pipeline bug? (2) If real, retrain the model on recent data that reflects the new reality. (3) Set up automated alerts so this detection runs periodically (daily/weekly) and notifies the team.

### Q5: What's the difference between data drift and concept drift?
**A:** Data drift = input distributions change (e.g., MonthlyCharges goes up). Concept drift = the relationship between inputs and output changes (e.g., high charges used to cause churn, but now customers are happy paying more because of new premium features). Data drift is easier to detect. Concept drift requires monitoring actual model performance over time.

### Q6: How would you run this in production?
**A:** Schedule the drift detection script to run daily/weekly (via cron job, Airflow, or a Lambda function). Store reference data statistics. Compare each batch of new predictions against the reference. Alert via email/Slack if drift exceeds the threshold. Log results to a monitoring dashboard.

### Q7: Why did you simulate drift instead of using real data?
**A:** We don't have a live production system generating real data. So I simulated realistic scenarios (price increase, tenure shift, fiber optic push) to demonstrate the detection capability. In production, you'd replace the simulation with actual API request logs.

### Q8: What threshold did you use and why?
**A:** P-value < 0.05 per feature (standard statistical significance). For overall dataset drift, I used 30% of features drifted — meaning if more than a third of features shift, the whole dataset is considered drifted. These are common starting points; in practice, you'd tune them based on business impact.

### Q9: Can drift detection give false alarms?
**A:** Yes. Statistical noise can trigger alerts, especially with small sample sizes. To reduce false alarms: (1) use a larger sample of production data, (2) run tests over multiple time windows, (3) focus on features the model actually relies on (top importance features) rather than all features.

### Q10: How does this connect to the rest of your MLOps pipeline?
**A:** It's the monitoring layer. The full flow is: Train model (MLflow) → Serve via API (FastAPI) → Test automatically (CI/CD) → Monitor for drift (this script). When drift is detected → retrain → re-test → redeploy. This closes the MLOps loop.

---

## Complete MLOps Pipeline (All Phases)

```
Phase 1-3: Data → EDA → Clean → Train → Tune → Save model
Phase 4:   Serve model via FastAPI REST API
Phase 5:   Auto-test on every push (GitHub Actions CI)
Phase 6:   Track experiments (MLflow)
Phase 7:   Monitor for data drift (KS test)
              ↓ (if drift detected)
           Retrain → loops back to Phase 3
```
