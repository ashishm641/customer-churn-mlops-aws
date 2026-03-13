"""
Model Monitoring — Data Drift Detection
=========================================
This script checks if new (incoming) data looks different from the
training data. If it does, the model might need retraining.

What it does:
1. Loads the training data (reference)
2. Simulates "new" production data with realistic drift
3. Compares distributions using the KS test (Kolmogorov-Smirnov)
4. Generates an HTML drift report you can open in the browser

In production, you'd replace the simulated data with actual incoming
customer data from your API logs.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Load the reference data (what the model was trained on)
# ---------------------------------------------------------------------------
print("Loading reference (training) data...")
df = pd.read_csv("data/processed/telco_churn_clean.csv")
reference_data = df.drop("Churn", axis=1)

print(f"Reference data: {reference_data.shape[0]} rows, {reference_data.shape[1]} features")

# ---------------------------------------------------------------------------
# 2. Create simulated "new" production data WITH drift
# ---------------------------------------------------------------------------
# In real life, this would be actual customer data from your API logs.
# Here we simulate realistic changes that could happen over time:
#   - Monthly charges went up (company raised prices)
#   - Tenure shifted (more new customers signed up)
#   - More fiber optic customers (company pushed fiber plans)

print("\nSimulating production data with drift...")
np.random.seed(42)
production_data = reference_data.copy()

# Drift 1: Monthly charges increased by ~15% (company raised prices)
production_data["MonthlyCharges"] = production_data["MonthlyCharges"] * 1.15 + np.random.normal(0, 3, len(production_data))

# Drift 2: TotalCharges also shifts (because it depends on MonthlyCharges)
production_data["TotalCharges"] = production_data["TotalCharges"] * 1.10 + np.random.normal(0, 50, len(production_data))

# Drift 3: More new customers (tenure shifts toward lower values)
production_data["tenure"] = np.maximum(1, production_data["tenure"] - np.random.randint(0, 12, len(production_data)))

# Drift 4: More fiber optic customers (company pushed fiber plans)
fiber_flip = np.random.random(len(production_data)) < 0.15
production_data.loc[fiber_flip, "InternetService_Fiber optic"] = 1
production_data.loc[fiber_flip, "InternetService_No"] = 0

print(f"Production data: {production_data.shape[0]} rows, {production_data.shape[1]} features")

# ---------------------------------------------------------------------------
# 3. Run drift detection using KS test
# ---------------------------------------------------------------------------
# The KS (Kolmogorov-Smirnov) test compares two distributions.
# If p-value < 0.05, the distributions are significantly different = DRIFT.
print("\nRunning drift detection (KS test per feature)...")

THRESHOLD = 0.05  # p-value below this = drift detected
results = []

for col in reference_data.columns:
    ref_values = reference_data[col].values
    prod_values = production_data[col].values

    ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)
    drifted = p_value < THRESHOLD

    results.append({
        "feature": col,
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "drifted": drifted,
    })

results_df = pd.DataFrame(results)
drifted_features = results_df[results_df["drifted"] == True]
n_drifted = len(drifted_features)
n_total = len(results_df)
drift_share = n_drifted / n_total

# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"DRIFT SUMMARY")
print(f"{'='*60}")
print(f"Features analyzed:  {n_total}")
print(f"Features drifted:   {n_drifted}")
print(f"Drift share:        {drift_share:.1%}")
print(f"Dataset drift:      {'YES - RETRAIN NEEDED' if drift_share > 0.3 else 'No - model is fine'}")
print(f"{'='*60}")

if n_drifted > 0:
    print(f"\nDrifted features:")
    print(f"{'Feature':<45} {'KS Stat':>10} {'P-Value':>12}")
    print(f"{'-'*67}")
    for _, row in drifted_features.iterrows():
        print(f"{row['feature']:<45} {row['ks_statistic']:>10.4f} {row['p_value']:>12.6f}")

if drift_share > 0.3:
    print("\nACTION NEEDED: The incoming data has shifted significantly.")
    print("The model may be making unreliable predictions.")
    print("Consider retraining with recent data.")
else:
    print("\nAll clear — data looks similar to training data.")
    print("Model predictions should still be reliable.")

# ---------------------------------------------------------------------------
# 5. Generate HTML report
# ---------------------------------------------------------------------------
report_path = Path("reports")
report_path.mkdir(exist_ok=True)
output_file = report_path / "data_drift_report.html"

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .alert {{ background: #ffe0e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff4444; margin: 20px 0; }}
        .ok {{ background: #e0ffe0; padding: 15px; border-radius: 8px; border-left: 4px solid #44ff44; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #4a90d9; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f9f9f9; }}
        .drifted {{ color: #cc0000; font-weight: bold; }}
        .stable {{ color: #009900; }}
    </style>
</head>
<body>
    <h1>Data Drift Report</h1>
    <p>Comparing training data (reference) vs production data (current)</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Features analyzed:</strong> {n_total}</p>
        <p><strong>Features drifted:</strong> {n_drifted} ({drift_share:.1%})</p>
        <p><strong>Detection method:</strong> Kolmogorov-Smirnov test (threshold: p &lt; {THRESHOLD})</p>
    </div>

    <div class="{'alert' if drift_share > 0.3 else 'ok'}">
        <strong>{'DRIFT DETECTED — Model retraining recommended' if drift_share > 0.3 else 'NO SIGNIFICANT DRIFT — Model is still reliable'}</strong>
    </div>

    <h2>Feature Details</h2>
    <table>
        <tr><th>Feature</th><th>KS Statistic</th><th>P-Value</th><th>Status</th></tr>
"""

for _, row in results_df.sort_values("p_value").iterrows():
    status_class = "drifted" if row["drifted"] else "stable"
    status_text = "DRIFTED" if row["drifted"] else "Stable"
    html_content += f'        <tr><td>{row["feature"]}</td><td>{row["ks_statistic"]:.4f}</td><td>{row["p_value"]:.6f}</td><td class="{status_class}">{status_text}</td></tr>\n'

html_content += """    </table>

    <div class="summary" style="margin-top: 30px;">
        <h2>What This Means</h2>
        <p><strong>KS Statistic:</strong> Measures how different two distributions are (0 = identical, 1 = completely different)</p>
        <p><strong>P-Value:</strong> Probability that the difference is due to random chance. Below 0.05 = statistically significant drift.</p>
        <p><strong>Drifted:</strong> The feature's distribution in production data is significantly different from training data.</p>
    </div>
</body>
</html>"""

with open(output_file, "w") as f:
    f.write(html_content)

print(f"\nFull HTML report saved to: {output_file.absolute()}")
print("Open it in your browser to see the detailed breakdown.")
