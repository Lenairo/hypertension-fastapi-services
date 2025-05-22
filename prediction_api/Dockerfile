FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files to avoid container bloat
COPY prediction_api.py .
COPY models ./models
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "prediction_api:app", "--host", "0.0.0.0", "--port", "8000"]
