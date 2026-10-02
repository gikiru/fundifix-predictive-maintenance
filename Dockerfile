# FundiFix Predictive Maintenance — Data Pipeline Container
# Build: docker build -t fundifix-pipeline .
# Run:   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/logs:/app/logs fundifix-pipeline

FROM python:3.11-slim

LABEL maintainer="gikiru"
LABEL project="fundifix-predictive-maintenance"
LABEL description="Data pipeline for rural water point failure prediction"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create required directories
RUN mkdir -p data/raw data/processed logs

# Default command: run the full pipeline
CMD ["python", "src/data/pipeline.py"]
