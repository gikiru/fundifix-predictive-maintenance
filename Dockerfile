# FundiFix pipeline and MLOps image: data pipeline, training, monitoring, retraining, MLflow.
# Build: docker build -t fundifix-pipeline .
# Run the pipeline alone:
#   docker run --rm -v ./data:/app/data -v ./logs:/app/logs -v ./reports:/app/reports fundifix-pipeline
# Normally started through docker-compose.yml.
FROM python:3.12-slim

LABEL maintainer="gikiru" project="fundifix-predictive-maintenance"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY api/ api/
RUN mkdir -p data/raw data/processed logs reports models app/data \
 && mkdir -p /app/store && chown 1000:1000 /app/store   # shared with the API container (user 1000)

CMD ["python", "src/data/pipeline.py"]
