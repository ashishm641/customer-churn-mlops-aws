# Phase 4 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

In Phase 3, we saved the best model as `best_model.pkl`. But it was just sitting in a file — nobody was using it.

Phase 4 asked: **"How do we let others use this model?"**

We built a **REST API using FastAPI** — a small web server that:
1. Loads the trained model once at startup
2. Accepts customer data as JSON via a POST request
3. Returns a churn prediction (0 or 1) and probability (e.g. 0.839)

We also created a **Dockerfile** to package everything into a portable container that can run anywhere.

---

## What We Built

### Files Created

| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI app with `/health` and `/predict` endpoints |
| `Dockerfile` | Recipe to package the app into a Docker container |
| `requirements-api.txt` | Lightweight dependencies for the container (no jupyter/matplotlib bloat) |

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if server is alive and model is loaded |
| `/predict` | POST | Send customer data, get churn prediction back |
| `/docs` | GET | Auto-generated Swagger UI for testing (built into FastAPI) |

---

## How It Works (Step by Step)

```
1. Server starts → loads best_model.pkl into memory (once)
2. Someone sends POST /predict with customer JSON
3. FastAPI validates the data (correct fields, correct types)
4. Data gets converted to numpy array (same format model was trained on)
5. model.predict() → 0 or 1
6. model.predict_proba() → probability (e.g. 0.839)
7. Returns JSON response with prediction + probability + message
```

---

## Key Concepts Explained

### What is an API?

A way for software to talk to other software. Instead of a human opening a notebook and running cells, a dashboard or mobile app sends a request to a URL and gets a response.

**Analogy:** A restaurant menu. You (the app) place an order (send a request), the kitchen (API) prepares it, and you get your food (response). You don't need to know how the kitchen works.

### What is FastAPI?

A Python web framework for building APIs. We chose it because:
- Very fast (one of the fastest Python frameworks)
- Auto-generates interactive docs at `/docs`
- Auto-validates input data using Pydantic models
- Simple syntax — just decorators like `@app.get()` and `@app.post()`

### What is Pydantic / BaseModel?

A "form template" that defines what data is required and what types each field should be.

```python
class CustomerData(BaseModel):
    tenure: float       # must be a number
    Partner: int        # must be 0 or 1
```

If someone sends `tenure: "hello"`, FastAPI automatically rejects it with an error. No manual validation needed.

### GET vs POST — When to Use Which?

| GET | POST |
|-----|------|
| Retrieve/read information | Send data for processing |
| No data in request body | Data sent in request body |
| Example: `/health` — just checking status | Example: `/predict` — sending 30 fields |
| Can be bookmarked/cached | Not cached |

### What is Docker?

A tool that packages your app + all its dependencies into a **container** — a lightweight, portable box that runs the same everywhere.

| Without Docker | With Docker |
|---|---|
| "Works on my machine" | Works on ANY machine |
| Install Python, packages, copy files | Just `docker run` |
| Different OS = different problems | Same behavior everywhere |

### What is the Dockerfile?

A recipe file with step-by-step instructions to build the container:
1. Start with Python 3.11
2. Install packages from requirements-api.txt
3. Copy the model file and API code
4. Run uvicorn (the web server)

### What is Uvicorn?

The actual web server that runs our FastAPI app. FastAPI is the framework (the code), uvicorn is the engine that listens for requests and handles them.

**Analogy:** FastAPI = the recipe, Uvicorn = the chef who follows the recipe and serves the food.

---

## Test Results

### High-Risk Customer (month-to-month, 2 months, $85.50, no tech support)
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.839,
  "message": "This customer is likely to churn."
}
```

### Low-Risk Customer (two-year contract, 60 months, $45, has tech support)
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.112,
  "message": "This customer is likely to stay."
}
```

Both match what we'd expect from our EDA in Phase 1.

---

## Interview Questions & Answers

### Q1: Why did you build an API instead of using a notebook?
**A:** Notebooks are for development and experimentation. In production, other systems (dashboards, mobile apps, automated pipelines) need to call the model programmatically. An API provides a standard interface — send JSON in, get JSON out — that any application can use.

### Q2: Why FastAPI and not Flask?
**A:** FastAPI is faster (async support), auto-generates documentation, and has built-in input validation via Pydantic. Flask works too, but requires more manual setup for validation and docs.

### Q3: What happens if someone sends invalid data to your API?
**A:** FastAPI automatically rejects it with a 422 Validation Error and tells the caller exactly which field is wrong and what type was expected. This is handled by the Pydantic model — I didn't write validation code manually.

### Q4: Why did you create a separate requirements-api.txt?
**A:** The Docker container only needs what the API uses (numpy, scikit-learn, fastapi, uvicorn). Including jupyter, matplotlib, and seaborn would make the container much larger for no reason. Smaller container = faster builds, faster deployments, less attack surface.

### Q5: Why load the model at startup instead of on every request?
**A:** Loading a pickle file from disk takes time. If we did it on every request, each prediction would be slower. Loading once at startup means the model sits in memory and predictions are instant.

### Q6: What is the /health endpoint for?
**A:** Health checks. In production, orchestrators like AWS ECS or Kubernetes ping this endpoint every few seconds. If it stops responding, they automatically restart the container. It's a standard practice for production services.

### Q7: Why use POST for predictions instead of GET?
**A:** Two reasons: (1) We're sending 30 fields of data — GET requests put data in the URL which has length limits and looks messy. (2) Semantically, POST means "process this data" while GET means "give me information." Predictions are a processing action.

### Q8: What does Docker give you that just running Python doesn't?
**A:** Reproducibility and portability. On my machine I have Python 3.14, specific package versions, specific OS settings. Docker packages the exact Python version, exact packages, and exact code into one container that runs identically everywhere — my laptop, a teammate's laptop, or AWS.

### Q9: How would you handle multiple prediction requests at the same time?
**A:** Uvicorn supports running multiple worker processes. In production, you'd run `uvicorn api.main:app --workers 4` to handle concurrent requests. FastAPI also supports async endpoints for I/O-bound work.

### Q10: What would you improve about this API?
**A:** Several things for production:
- Add logging (track each prediction for monitoring)
- Add input range validation (tenure shouldn't be negative)
- Add authentication (API key or JWT so not everyone can call it)
- Return feature importance / explanation for each prediction (model interpretability)
- Add a batch prediction endpoint for multiple customers at once

---

## What's Next?

Phase 5 — Deploy to AWS using containers (ECR + ECS) so the API is accessible over the internet, not just localhost.
