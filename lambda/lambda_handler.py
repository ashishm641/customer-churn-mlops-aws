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

    # --- Root / default — serve the web UI ---
    return html_response(200, build_html_page())


def response(status_code, body):
    """Helper to format Lambda JSON response for API Gateway."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def html_response(status_code, html_body):
    """Helper to return an HTML page from Lambda."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "text/html",
        },
        "body": html_body,
    }


def build_html_page():
    """Build the prediction web UI as an HTML string."""

    # Friendly labels for the 30 features, grouped for UX
    form_sections = [
        ("Customer Info", [
            ("gender", "Gender", "0 = Female, 1 = Male"),
            ("SeniorCitizen", "Senior Citizen", "0 = No, 1 = Yes"),
            ("Partner", "Has Partner", "0 = No, 1 = Yes"),
            ("Dependents", "Has Dependents", "0 = No, 1 = Yes"),
        ]),
        ("Account Info", [
            ("tenure", "Tenure (months)", "Number of months as customer"),
            ("MonthlyCharges", "Monthly Charges ($)", "e.g. 70.35"),
            ("TotalCharges", "Total Charges ($)", "e.g. 1397.47"),
            ("PhoneService", "Phone Service", "0 = No, 1 = Yes"),
            ("PaperlessBilling", "Paperless Billing", "0 = No, 1 = Yes"),
        ]),
        ("Phone", [
            ("MultipleLines_No_phone_service", "No Phone Service", "1 if no phone service"),
            ("MultipleLines_Yes", "Multiple Lines", "1 = Yes"),
        ]),
        ("Internet Service", [
            ("InternetService_Fiber_optic", "Fiber Optic", "1 = Yes"),
            ("InternetService_No", "No Internet", "1 if no internet service"),
        ]),
        ("Online Services", [
            ("OnlineSecurity_No_internet_service", "Online Security — No Internet", "1 if no internet"),
            ("OnlineSecurity_Yes", "Online Security", "1 = Yes"),
            ("OnlineBackup_No_internet_service", "Online Backup — No Internet", "1 if no internet"),
            ("OnlineBackup_Yes", "Online Backup", "1 = Yes"),
            ("DeviceProtection_No_internet_service", "Device Protection — No Internet", "1 if no internet"),
            ("DeviceProtection_Yes", "Device Protection", "1 = Yes"),
            ("TechSupport_No_internet_service", "Tech Support — No Internet", "1 if no internet"),
            ("TechSupport_Yes", "Tech Support", "1 = Yes"),
        ]),
        ("Streaming", [
            ("StreamingTV_No_internet_service", "Streaming TV — No Internet", "1 if no internet"),
            ("StreamingTV_Yes", "Streaming TV", "1 = Yes"),
            ("StreamingMovies_No_internet_service", "Streaming Movies — No Internet", "1 if no internet"),
            ("StreamingMovies_Yes", "Streaming Movies", "1 = Yes"),
        ]),
        ("Contract & Payment", [
            ("Contract_One_year", "One Year Contract", "1 = Yes"),
            ("Contract_Two_year", "Two Year Contract", "1 = Yes"),
            ("PaymentMethod_Credit_card_automatic", "Credit Card (Auto)", "1 = Yes"),
            ("PaymentMethod_Electronic_check", "Electronic Check", "1 = Yes"),
            ("PaymentMethod_Mailed_check", "Mailed Check", "1 = Yes"),
        ]),
    ]

    # Build form fields HTML
    fields_html = ""
    for section_title, fields in form_sections:
        fields_html += f'<div class="section"><h3>{section_title}</h3><div class="grid">'
        for key, label, hint in fields:
            default = "0"
            step = "any" if key in ("MonthlyCharges", "TotalCharges") else "1"
            fields_html += (
                f'<div class="field">'
                f'<label for="{key}">{label}</label>'
                f'<input type="number" id="{key}" name="{key}" value="{default}" step="{step}" />'
                f'<small>{hint}</small>'
                f'</div>'
            )
        fields_html += '</div></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Customer Churn Predictor</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #333; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ text-align: center; color: #1a73e8; margin: 20px 0 5px; }}
        .subtitle {{ text-align: center; color: #666; margin-bottom: 25px; font-size: 14px; }}
        .section {{ background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 15px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .section h3 {{ color: #1a73e8; margin-bottom: 12px; border-bottom: 2px solid #e8f0fe; padding-bottom: 6px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
        .field {{ display: flex; flex-direction: column; }}
        .field label {{ font-weight: 600; font-size: 13px; margin-bottom: 3px; }}
        .field input {{ padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
        .field input:focus {{ outline: none; border-color: #1a73e8; box-shadow: 0 0 0 2px #e8f0fe; }}
        .field small {{ color: #999; font-size: 11px; margin-top: 2px; }}
        .btn-row {{ text-align: center; margin: 20px 0; }}
        .btn {{ background: #1a73e8; color: #fff; border: none; padding: 14px 48px; font-size: 16px;
                border-radius: 8px; cursor: pointer; font-weight: 600; }}
        .btn:hover {{ background: #1557b0; }}
        .btn:disabled {{ background: #94c2f8; cursor: wait; }}
        #result {{ display: none; margin: 20px auto; max-width: 500px; padding: 25px; border-radius: 10px;
                   text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .churn {{ background: #fce8e6; border: 2px solid #ea4335; }}
        .stay {{ background: #e6f4ea; border: 2px solid #34a853; }}
        .result-icon {{ font-size: 48px; margin-bottom: 10px; }}
        .result-label {{ font-size: 22px; font-weight: 700; }}
        .result-prob {{ font-size: 16px; color: #555; margin-top: 8px; }}
        .error {{ background: #fff3e0; border: 2px solid #ff9800; }}
        .prefill {{ text-align: center; margin-bottom: 15px; }}
        .prefill button {{ background: #fff; border: 1px solid #1a73e8; color: #1a73e8; padding: 6px 16px;
                           border-radius: 6px; cursor: pointer; margin: 0 5px; font-size: 13px; }}
        .prefill button:hover {{ background: #e8f0fe; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Customer Churn Predictor</h1>
    <p class="subtitle">Enter customer details below and click Predict to check if they will churn</p>

    <div class="prefill">
        <button onclick="fillHighRisk()">Fill High-Risk Example</button>
        <button onclick="fillLowRisk()">Fill Low-Risk Example</button>
        <button onclick="resetForm()">Reset All</button>
    </div>

    <form id="churnForm">
        {fields_html}
        <div class="btn-row">
            <button type="submit" class="btn" id="predictBtn">Predict Churn</button>
        </div>
    </form>

    <div id="result">
        <div class="result-icon" id="resultIcon"></div>
        <div class="result-label" id="resultLabel"></div>
        <div class="result-prob" id="resultProb"></div>
    </div>
</div>

<script>
    const FEATURE_KEYS = {json.dumps(FEATURE_KEYS)};

    document.getElementById('churnForm').addEventListener('submit', async function(e) {{
        e.preventDefault();
        const btn = document.getElementById('predictBtn');
        btn.disabled = true;
        btn.textContent = 'Predicting...';

        const payload = {{}};
        FEATURE_KEYS.forEach(key => {{
            payload[key] = parseFloat(document.getElementById(key).value) || 0;
        }});

        try {{
            const resp = await fetch(window.location.origin + '/predict', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload)
            }});
            const data = await resp.json();

            const box = document.getElementById('result');
            const icon = document.getElementById('resultIcon');
            const label = document.getElementById('resultLabel');
            const prob = document.getElementById('resultProb');

            box.style.display = 'block';
            box.className = '';

            if (data.error) {{
                box.classList.add('error');
                icon.textContent = '⚠️';
                label.textContent = 'Error';
                prob.textContent = data.error;
            }} else if (data.churn_prediction === 1) {{
                box.classList.add('churn');
                icon.textContent = '🚨';
                label.textContent = 'Likely to CHURN';
                prob.textContent = 'Churn probability: ' + (data.churn_probability * 100).toFixed(1) + '%';
            }} else {{
                box.classList.add('stay');
                icon.textContent = '✅';
                label.textContent = 'Likely to STAY';
                prob.textContent = 'Churn probability: ' + (data.churn_probability * 100).toFixed(1) + '%';
            }}

            box.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }} catch (err) {{
            alert('Request failed: ' + err.message);
        }} finally {{
            btn.disabled = false;
            btn.textContent = 'Predict Churn';
        }}
    }});

    function fillHighRisk() {{
        const vals = {{gender:1,SeniorCitizen:0,Partner:0,Dependents:0,tenure:2,PhoneService:1,
            PaperlessBilling:1,MonthlyCharges:80,TotalCharges:160,
            MultipleLines_No_phone_service:0,MultipleLines_Yes:1,
            InternetService_Fiber_optic:1,InternetService_No:0,
            OnlineSecurity_No_internet_service:0,OnlineSecurity_Yes:0,
            OnlineBackup_No_internet_service:0,OnlineBackup_Yes:0,
            DeviceProtection_No_internet_service:0,DeviceProtection_Yes:0,
            TechSupport_No_internet_service:0,TechSupport_Yes:0,
            StreamingTV_No_internet_service:0,StreamingTV_Yes:0,
            StreamingMovies_No_internet_service:0,StreamingMovies_Yes:0,
            Contract_One_year:0,Contract_Two_year:0,
            PaymentMethod_Credit_card_automatic:0,
            PaymentMethod_Electronic_check:1,PaymentMethod_Mailed_check:0}};
        Object.entries(vals).forEach(([k,v]) => document.getElementById(k).value = v);
    }}

    function fillLowRisk() {{
        const vals = {{gender:0,SeniorCitizen:0,Partner:1,Dependents:1,tenure:60,PhoneService:1,
            PaperlessBilling:0,MonthlyCharges:25,TotalCharges:1500,
            MultipleLines_No_phone_service:0,MultipleLines_Yes:0,
            InternetService_Fiber_optic:0,InternetService_No:0,
            OnlineSecurity_No_internet_service:0,OnlineSecurity_Yes:1,
            OnlineBackup_No_internet_service:0,OnlineBackup_Yes:1,
            DeviceProtection_No_internet_service:0,DeviceProtection_Yes:1,
            TechSupport_No_internet_service:0,TechSupport_Yes:1,
            StreamingTV_No_internet_service:0,StreamingTV_Yes:0,
            StreamingMovies_No_internet_service:0,StreamingMovies_Yes:0,
            Contract_One_year:0,Contract_Two_year:1,
            PaymentMethod_Credit_card_automatic:1,
            PaymentMethod_Electronic_check:0,PaymentMethod_Mailed_check:0}};
        Object.entries(vals).forEach(([k,v]) => document.getElementById(k).value = v);
    }}

    function resetForm() {{
        FEATURE_KEYS.forEach(key => document.getElementById(key).value = 0);
        document.getElementById('result').style.display = 'none';
    }}
</script>
</body>
</html>"""
