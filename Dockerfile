FROM python:3.11-slim

WORKDIR /app

# Install only what the API needs (not jupyter, matplotlib, etc.)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the model and API code
COPY models/best_model.pkl models/best_model.pkl
COPY api/main.py api/main.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
