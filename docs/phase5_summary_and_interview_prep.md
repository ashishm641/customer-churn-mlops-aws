# Phase 5 — Summary & Interview Prep

---

## What We Did (Plain English Summary)

In Phase 4, we built a working API. But there was no safety net — if someone pushed broken code, nobody would know until things crashed.

Phase 5 asked: **"How do we catch bugs automatically?"**

We did two things:
1. Wrote **5 automated tests** that check if the API works correctly
2. Set up **GitHub Actions CI/CD** so these tests run automatically on every push

Now, every time code is pushed to the main branch, GitHub spins up a fresh machine, installs everything, runs the tests, and reports pass/fail — all in about 39 seconds.

---

## What We Built

### Files Created

| File | Purpose |
|------|---------|
| `tests/test_api.py` | 5 automated tests for the API |
| `.github/workflows/ci.yml` | Instructions telling GitHub when and how to run tests |

### The 5 Tests

| Test | What It Checks |
|------|---------------|
| `test_health_check` | Is the server alive? Is the model loaded? |
| `test_predict_returns_correct_fields` | Does the response have prediction, probability, and message? |
| `test_high_risk_customer_predicts_churn` | Does a classic churner profile get predicted as churn? |
| `test_low_risk_customer_predicts_stay` | Does a loyal customer profile get predicted as stay? |
| `test_missing_fields_returns_422` | Does incomplete data get properly rejected? |

### The CI Pipeline (8 Steps on GitHub)

| Step | Who Creates It | What It Does |
|------|---------------|-------------|
| Set up job | GitHub (automatic) | Creates a fresh Ubuntu machine |
| Checkout code | Our YAML file | Downloads our repo code |
| Set up Python | Our YAML file | Installs Python 3.11 |
| Install dependencies | Our YAML file | pip install all packages |
| Run tests | Our YAML file | Runs pytest with all 5 tests |
| Post Set up Python | GitHub (automatic) | Cleanup |
| Post Checkout code | GitHub (automatic) | Cleanup |
| Complete job | GitHub (automatic) | Reports pass/fail |

We wrote 4 steps. GitHub added 4 automatically.

---

## Key Concepts Explained

### What is CI/CD?

- **CI (Continuous Integration)** = Every time you push code, tests run automatically to check if anything broke
- **CD (Continuous Deployment)** = If tests pass, automatically deploy to production

We did CI. CD would be the next step (auto-deploy to AWS when tests pass).

**Analogy:** CI is like a spell-checker that runs every time you save a document. You don't have to remember to check — it happens automatically.

### What is GitHub Actions?

GitHub's built-in CI/CD tool. It reads YAML files from `.github/workflows/` and runs them when triggered (on push, on pull request, on schedule, etc.). It's free for public repos.

### What is a YAML Workflow File?

A recipe that tells GitHub:
- **When** to run (on push to main)
- **Where** to run (ubuntu-latest)
- **What** to run (install Python, install packages, run tests)

### What is pytest?

A Python testing framework. You write functions that start with `test_`, and pytest finds and runs them all. If any `assert` statement fails, the test fails.

```python
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200    # if not 200, test FAILS
```

### What is TestClient?

A fake HTTP client from FastAPI that lets you test your API **without starting a real server**. It calls your endpoints directly in memory — faster and simpler.

### What is a 422 Error?

"Unprocessable Entity" — the server understood your request format but the data is invalid (missing fields, wrong types). FastAPI returns this automatically when Pydantic validation fails.

---

## How It All Connects

```
Developer pushes code
    ↓
GitHub detects .github/workflows/ci.yml
    ↓
Spins up fresh Ubuntu machine
    ↓
Installs Python + packages
    ↓
Runs pytest (5 tests)
    ↓
All pass → Green ✅       Any fail → Red ❌ + email alert
```

---

## Interview Questions & Answers

### Q1: What is CI/CD and why did you use it?
**A:** CI (Continuous Integration) means running automated tests every time code is pushed. I used GitHub Actions to automatically test my API on every push to main. This catches bugs early — before they reach production. It's a standard practice in any professional development workflow.

### Q2: What tests did you write and why those specifically?
**A:** Five tests covering different scenarios: (1) health check works, (2) prediction returns correct response format, (3) high-risk customer gets predicted as churn, (4) low-risk customer gets predicted as stay, (5) invalid input gets properly rejected. These cover the happy path, edge cases, and error handling.

### Q3: Why test with a high-risk AND low-risk customer?
**A:** To verify the model works in both directions. If I only tested one type, the model could be returning the same answer for everyone and I wouldn't catch it. Testing both extremes confirms the model actually differentiates between churners and non-churners.

### Q4: What happens if a test fails when you push?
**A:** GitHub shows a red X on the commit. I get an email notification. The code is still pushed (it doesn't block the push), but the team knows something is broken. In a production setup, you'd configure it to block deployment until tests pass.

### Q5: Why use GitHub Actions instead of running tests manually?
**A:** Humans forget. Automation doesn't. If I run tests manually, one day I'll forget and push broken code. With CI, every single push is tested — no exceptions. It also tests on a clean machine, so "it works on my machine" problems get caught.

### Q6: What is the difference between testing locally and testing in CI?
**A:** Locally, I test on MY machine with MY Python version and MY installed packages. CI tests on a fresh machine with nothing pre-installed. This catches issues like: missing dependencies in requirements.txt, code that only works on Windows but fails on Linux, etc.

### Q7: Why did you use FastAPI's TestClient instead of hitting the real server?
**A:** TestClient calls the API directly in memory — no need to start a server, no network involved. It's faster, more reliable, and easier to run in CI. If the test depends on a running server, it becomes fragile (what if the port is in use? what if the server is slow to start?).

### Q8: What does the 422 status code test verify?
**A:** It verifies input validation works. If someone sends incomplete data (only 2 fields out of 30), the API should reject it with a clear error — not crash, not return garbage. This is important for production reliability.

### Q9: How would you add a new test?
**A:** Just write a new function starting with `test_` in `tests/test_api.py`. Pytest auto-discovers it. On next push, GitHub Actions will run it along with the existing tests. No configuration needed.

### Q10: What would you add to this CI pipeline in a real production setup?
**A:** Several things:
- **Code linting** (flake8/ruff) — check code style automatically
- **Type checking** (mypy) — catch type errors
- **Docker build** — verify the container builds successfully
- **CD step** — auto-deploy to AWS if all tests pass
- **Test coverage** — measure what percentage of code is tested
- **Multiple Python versions** — test on 3.10, 3.11, 3.12

---

## What's Next?

Phase 6 options:
- **MLflow** — Track experiments, compare model runs, log metrics
- **AWS Deployment** — Deploy the API to the cloud (ECR + ECS)
- **Model Monitoring** — Detect when model performance degrades over time
