# Phase 8: AWS Lambda Deployment — Summary & Interview Prep

## What We Did

We deployed our churn prediction model as a **serverless API** on AWS using **Lambda + API Gateway**.

### Architecture
```
User → HTTPS Request → API Gateway → Lambda Function → ML Model → Response
```

### Steps Taken
1. **Created Lambda handler** — Rewrote the FastAPI logic as a plain Python function (`lambda_handler`) that receives an event dictionary from API Gateway and returns a JSON response.
2. **Packaged dependencies** — Bundled numpy, scikit-learn, scipy, joblib (Linux x86_64 binaries for Python 3.11) + model file + handler into a deployment zip (~60MB).
3. **Created IAM role** — `churn-lambda-role` with `AWSLambdaBasicExecutionRole` policy (allows the function to write logs to CloudWatch).
4. **Uploaded to S3** — Zip exceeded the 50MB direct upload limit, so we uploaded to an S3 bucket first.
5. **Created Lambda function** — 512MB memory, 30s timeout, Python 3.11 runtime.
6. **Created HTTP API Gateway** — Routes all requests to the Lambda function, provides the public HTTPS URL.
7. **Tested live endpoints** — Both `/health` (GET) and `/predict` (POST) work on the public URL.

### Live API
- **Base URL**: `https://rdbbxjfr8a.execute-api.us-east-1.amazonaws.com`
- **Health**: `GET /health` → `{"status": "healthy", "model_loaded": true}`
- **Predict**: `POST /predict` with 30 features → `{"churn_prediction": 1, "churn_probability": 0.833, "message": "This customer is likely to churn."}`

### Why Serverless?
- **Zero cost at idle** — Pay only when requests come in (~$0.20 per million requests).
- **No server management** — No EC2 instances to patch, monitor, or scale.
- **Auto-scaling** — Handles 1 request or 1000 concurrent requests automatically.
- **Built-in HTTPS** — API Gateway provides TLS certificates out of the box.

### Key Decisions
| Decision | Why |
|----------|-----|
| Lambda over EC2/ECS | Zero idle cost, no infra to manage |
| S3 upload over direct | Zip was 60MB, over the 50MB direct upload limit |
| Removed scipy initially | Tried to reduce size, but sklearn needs scipy at import time |
| 512MB memory | ML model inference needs more RAM than the default 128MB |
| HTTP API (v2) over REST API (v1) | Cheaper, faster, simpler for our use case |

---

## Interview Q&A

### Q1: Why did you choose AWS Lambda over EC2 or ECS?
**A:** Lambda is a serverless compute service — I only pay when the API receives requests. For a portfolio project (or low-traffic production API), this means **zero cost at idle**. EC2 would cost ~$8-15/month even when no one is using it. ECS is great for high-traffic but needs cluster management. Lambda was the right fit for a lightweight inference endpoint.

### Q2: What is AWS Lambda? Explain it simply.
**A:** Lambda is like a vending machine — it only runs when someone presses a button (sends a request). You give AWS your code in a zip file, and AWS handles everything: servers, scaling, patching. You pay per request (fractions of a cent), not per hour. Your code runs, returns a result, and shuts down.

### Q3: What is API Gateway and why do you need it?
**A:** Lambda by itself doesn't have a public URL. API Gateway acts as the "front door" — it gives you an HTTPS URL, routes incoming requests to your Lambda function, handles SSL certificates, and can do rate limiting and authentication. Think of it as a receptionist that forwards calls to the right person (Lambda).

### Q4: How does the Lambda handler differ from your FastAPI app?
**A:** FastAPI runs as a **web server** (always on, listening for requests). Lambda receives a single **event dictionary** from API Gateway, processes it, and returns a response dictionary. There's no server loop. Key differences:
- No `@app.get()` decorators — I manually check the path/method from the event
- No Pydantic validation — I parse the JSON body myself
- The model loads at **module level** so it persists across warm invocations (Lambda reuses the container)

### Q5: What's a "cold start" in Lambda? How did you handle it?
**A:** A cold start happens when Lambda creates a new container for your function — it downloads your code, starts the runtime, and runs your initialization code. For our function, this means loading the ML model (~1MB pickle file) and importing numpy/sklearn. Cold starts add ~3-5 seconds. After that, the container stays warm for ~15 minutes, and subsequent requests take <1 second. To minimize cold starts, I:
- Kept the deployment package small (stripped unnecessary files)
- Load the model at module level (not inside the handler function)
- Chose 512MB memory (more memory = more CPU = faster initialization)

### Q6: Why did your deployment fail initially?
**A:** Two issues:
1. **Unzipped size limit** — Lambda has a 250MB limit for unzipped packages. My initial package with scipy was ~250MB+. I stripped `__pycache__`, test directories, and `.dist-info` folders to get it to 175MB.
2. **Missing scipy** — I first tried removing scipy to save space, but scikit-learn imports scipy at module level. The Lambda logs showed `No module named 'scipy'`. I had to include scipy and trim it differently.

### Q7: What is an IAM role and why does Lambda need one?
**A:** An IAM role is like a badge that gives an AWS service specific permissions. Lambda needs a role with `AWSLambdaBasicExecutionRole` policy so it can write logs to CloudWatch. Without it, Lambda can't even start. The "trust policy" says "only the Lambda service can wear this badge" — preventing other services from misusing the permissions.

### Q8: How would you update the model in production?
**A:** I would:
1. Train the new model and save the pickle file
2. Rebuild the deployment zip with the new model
3. Upload to S3 and run `aws lambda update-function-code`
4. Lambda atomically switches to the new code — zero downtime
5. In a real setup, I'd automate this in the CI/CD pipeline: model training → packaging → Lambda deployment, triggered by a new model being registered in MLflow.

### Q9: What are Lambda's key limitations for ML inference?
**A:** 
- **250MB unzipped package limit** — Can't use large libraries like TensorFlow/PyTorch directly (would need container images instead)
- **15-minute max timeout** — Fine for inference, not for training
- **Cold starts** — First request after idle period is slower
- **10MB response limit** — Fine for JSON predictions, not for returning large files
- **Stateless** — Can't store data between invocations (use DynamoDB/S3 for persistence)

### Q10: How would you make this production-ready?
**A:** Several improvements:
- **Custom domain** — Use Route 53 + ACM certificate for a clean URL like `api.mycompany.com/predict`
- **Authentication** — Add API keys or Cognito authorizer to restrict access
- **Logging & monitoring** — CloudWatch alarms on error rates, latency, and invocation count
- **Versioning** — Lambda aliases (e.g., `prod`, `staging`) pointing to specific versions
- **CI/CD deployment** — GitHub Actions deploys new Lambda code on merge to main
- **Input validation** — Validate feature ranges server-side to catch bad data
- **Container image** — If the package gets bigger, switch from zip to Docker container (supports up to 10GB)
