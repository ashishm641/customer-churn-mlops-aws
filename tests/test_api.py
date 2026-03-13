"""
Tests for the Customer Churn Prediction API.

These tests check:
1. Is the health endpoint working?
2. Can we make predictions?
3. Does a high-risk customer get predicted as churn?
4. Does a low-risk customer get predicted as stay?
5. Does the API reject bad input?
"""

from fastapi.testclient import TestClient
from api.main import app

# This creates a fake client that talks to our API without starting a real server
client = TestClient(app)


# ---- Test 1: Health check ----
# The simplest test. Just hit /health and check the response.
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ---- Test 2: Predict endpoint returns the right shape ----
# Send valid data, check we get prediction + probability + message back.
def test_predict_returns_correct_fields():
    customer = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 0,
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 0,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0,
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    data = response.json()
    # Check all three fields exist
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "message" in data
    # Prediction should be 0 or 1
    assert data["churn_prediction"] in [0, 1]
    # Probability should be between 0 and 1
    assert 0 <= data["churn_probability"] <= 1


# ---- Test 3: High-risk customer should be predicted as churn ----
# Month-to-month, 2 months tenure, high charges, no support = classic churner
def test_high_risk_customer_predicts_churn():
    high_risk = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 2,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 1,
        "InternetService_Fiber_optic": 1,
        "InternetService_No": 0,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 0,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 0,
        "PaymentMethod_Credit_card_automatic": 0,
        "PaymentMethod_Electronic_check": 1,
        "PaymentMethod_Mailed_check": 0,
    }
    response = client.post("/predict", json=high_risk)
    data = response.json()
    assert data["churn_prediction"] == 1
    assert data["churn_probability"] > 0.5


# ---- Test 4: Low-risk customer should be predicted as stay ----
# Two-year contract, 60 months tenure, low charges, has support = loyal customer
def test_low_risk_customer_predicts_stay():
    low_risk = {
        "gender": 0,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 1,
        "tenure": 60,
        "PhoneService": 1,
        "PaperlessBilling": 0,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 0,
        "InternetService_Fiber_optic": 0,
        "InternetService_No": 0,
        "OnlineSecurity_No_internet_service": 0,
        "OnlineSecurity_Yes": 1,
        "OnlineBackup_No_internet_service": 0,
        "OnlineBackup_Yes": 1,
        "DeviceProtection_No_internet_service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No_internet_service": 0,
        "TechSupport_Yes": 1,
        "StreamingTV_No_internet_service": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No_internet_service": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One_year": 0,
        "Contract_Two_year": 1,
        "PaymentMethod_Credit_card_automatic": 1,
        "PaymentMethod_Electronic_check": 0,
        "PaymentMethod_Mailed_check": 0,
    }
    response = client.post("/predict", json=low_risk)
    data = response.json()
    assert data["churn_prediction"] == 0
    assert data["churn_probability"] < 0.5


# ---- Test 5: Missing fields should return 422 error ----
# If someone sends incomplete data, the API should reject it
def test_missing_fields_returns_422():
    incomplete = {"gender": 1, "tenure": 12}  # missing 28 fields
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # Unprocessable Entity
