"""
Customer Churn Prediction API
Loads the trained model and serves predictions via REST endpoints.
"""

import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Load the trained model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Define the input schema — these are the 30 features the model expects
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    gender: int                                    # 0 = Female, 1 = Male
    SeniorCitizen: int                             # 0 or 1
    Partner: int                                   # 0 or 1
    Dependents: int                                # 0 or 1
    tenure: float                                  # months with the company
    PhoneService: int                              # 0 or 1
    PaperlessBilling: int                          # 0 or 1
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No_phone_service: int            # 0 or 1
    MultipleLines_Yes: int                         # 0 or 1
    InternetService_Fiber_optic: int               # 0 or 1
    InternetService_No: int                        # 0 or 1
    OnlineSecurity_No_internet_service: int        # 0 or 1
    OnlineSecurity_Yes: int                        # 0 or 1
    OnlineBackup_No_internet_service: int          # 0 or 1
    OnlineBackup_Yes: int                          # 0 or 1
    DeviceProtection_No_internet_service: int      # 0 or 1
    DeviceProtection_Yes: int                      # 0 or 1
    TechSupport_No_internet_service: int           # 0 or 1
    TechSupport_Yes: int                           # 0 or 1
    StreamingTV_No_internet_service: int           # 0 or 1
    StreamingTV_Yes: int                           # 0 or 1
    StreamingMovies_No_internet_service: int       # 0 or 1
    StreamingMovies_Yes: int                       # 0 or 1
    Contract_One_year: int                         # 0 or 1
    Contract_Two_year: int                         # 0 or 1
    PaymentMethod_Credit_card_automatic: int       # 0 or 1
    PaymentMethod_Electronic_check: int            # 0 or 1
    PaymentMethod_Mailed_check: int                # 0 or 1

# The exact column names the model was trained on (order matters!)
FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]

# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn based on their profile.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Check if the API is running and the model is loaded."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    """
    Accept customer data and return churn prediction + probability.
    """
    # Convert input to a numpy array in the correct feature order
    features = np.array([[
        customer.gender,
        customer.SeniorCitizen,
        customer.Partner,
        customer.Dependents,
        customer.tenure,
        customer.PhoneService,
        customer.PaperlessBilling,
        customer.MonthlyCharges,
        customer.TotalCharges,
        customer.MultipleLines_No_phone_service,
        customer.MultipleLines_Yes,
        customer.InternetService_Fiber_optic,
        customer.InternetService_No,
        customer.OnlineSecurity_No_internet_service,
        customer.OnlineSecurity_Yes,
        customer.OnlineBackup_No_internet_service,
        customer.OnlineBackup_Yes,
        customer.DeviceProtection_No_internet_service,
        customer.DeviceProtection_Yes,
        customer.TechSupport_No_internet_service,
        customer.TechSupport_Yes,
        customer.StreamingTV_No_internet_service,
        customer.StreamingTV_Yes,
        customer.StreamingMovies_No_internet_service,
        customer.StreamingMovies_Yes,
        customer.Contract_One_year,
        customer.Contract_Two_year,
        customer.PaymentMethod_Credit_card_automatic,
        customer.PaymentMethod_Electronic_check,
        customer.PaymentMethod_Mailed_check,
    ]])

    # Get prediction (0 or 1) and probability
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 3),
        "message": "This customer is likely to churn." if prediction == 1
                   else "This customer is likely to stay.",
    }
