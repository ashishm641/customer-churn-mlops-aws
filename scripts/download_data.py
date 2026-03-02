"""
download_data.py
----------------
PURPOSE: Download the Telco Customer Churn dataset to your local machine.

WHAT IS THIS DATASET?
    A telecom company tracked ~7,000 customers for one month.
    For each customer they recorded:
      - Demographics (age, gender, whether they have a partner/dependents)
      - Services they use (phone, internet, TV streaming, etc.)
      - Account info (how long they've been a customer, monthly charges, contract type)
      - Did they LEAVE (churn) that month? YES or NO

WHY THIS DATASET?
    - Small enough to run on a laptop (7,000 rows, 21 columns)
    - Clean enough to learn from (few missing values)
    - Real enough to understand business problems
    - Used in interviews at many companies

HOW TO RUN:
    python scripts/download_data.py
"""

import urllib.request
import os

# Where we want to save the file
SAVE_PATH = "data/raw/telco_churn.csv"

# Public URL of the dataset (IBM's sample dataset - commonly used for learning)
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

def download():
    # Make sure the folder exists
    os.makedirs("data/raw", exist_ok=True)

    # Don't re-download if already present
    if os.path.exists(SAVE_PATH):
        print(f"[INFO] File already exists: {SAVE_PATH}")
        print("[INFO] Delete it and re-run if you want a fresh copy.")
    else:
        print(f"[INFO] Downloading dataset from:\n       {DATA_URL}\n")
        urllib.request.urlretrieve(DATA_URL, SAVE_PATH)
        print(f"[OK]   Saved to: {SAVE_PATH}")

    # Show basic file info so you can confirm it worked
    size_kb = os.path.getsize(SAVE_PATH) / 1024
    print(f"[OK]   File size: {size_kb:.1f} KB")

    # Count rows (quick sanity check - no pandas needed)
    with open(SAVE_PATH, "r") as f:
        row_count = sum(1 for _ in f) - 1   # minus 1 for the header

    print(f"[OK]   Rows (customers): {row_count}")
    print()
    print("NEXT STEP:")
    print("  Open a Jupyter notebook and load this file with pandas.")
    print("  We'll do that together in Checkpoint 2.")

if __name__ == "__main__":
    download()
