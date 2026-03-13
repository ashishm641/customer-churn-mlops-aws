"""
AWS Lambda Handler for Customer Churn Prediction
==================================================
This is a lightweight version of our FastAPI app, adapted for Lambda.
Lambda doesn't run a web server — it receives an event and returns a response.
"""

import json
import pickle
import numpy as np
import os

# ---------------------------------------------------------------------------
# Load model at module level (reused across Lambda invocations)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# The 30 features in the correct order
FEATURE_KEYS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_No_phone_service", "MultipleLines_Yes",
    "InternetService_Fiber_optic", "InternetService_No",
    "OnlineSecurity_No_internet_service", "OnlineSecurity_Yes",
    "OnlineBackup_No_internet_service", "OnlineBackup_Yes",
    "DeviceProtection_No_internet_service", "DeviceProtection_Yes",
    "TechSupport_No_internet_service", "TechSupport_Yes",
    "StreamingTV_No_internet_service", "StreamingTV_Yes",
    "StreamingMovies_No_internet_service", "StreamingMovies_Yes",
    "Contract_One_year", "Contract_Two_year",
    "PaymentMethod_Credit_card_automatic",
    "PaymentMethod_Electronic_check", "PaymentMethod_Mailed_check",
]


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    Handles /health (GET) and /predict (POST) routes.
    """
    # Parse the route and method
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("path") or event.get("rawPath", "/")

    # --- Health check ---
    if path.endswith("/health"):
        return response(200, {"status": "healthy", "model_loaded": model is not None})

    # --- Predict ---
    if path.endswith("/predict") and http_method == "POST":
        try:
            # Parse body
            body = event.get("body", "{}")
            if isinstance(body, str):
                body = json.loads(body)

            # Validate all features present
            missing = [k for k in FEATURE_KEYS if k not in body]
            if missing:
                return response(422, {"error": f"Missing fields: {missing}"})

            # Build feature array
            features = np.array([[body[k] for k in FEATURE_KEYS]])

            # Predict
            prediction = int(model.predict(features)[0])
            probability = float(model.predict_proba(features)[0][1])

            return response(200, {
                "churn_prediction": prediction,
                "churn_probability": round(probability, 3),
                "message": "This customer is likely to churn." if prediction == 1
                           else "This customer is likely to stay.",
            })

        except json.JSONDecodeError:
            return response(400, {"error": "Invalid JSON in request body"})
        except Exception as e:
            return response(500, {"error": str(e)})

    # --- Root / default ---
    return response(200, {
        "service": "Customer Churn Prediction API",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Predict churn for a customer",
        }
    })


def response(status_code, body):
    """Helper to format Lambda response for API Gateway."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
