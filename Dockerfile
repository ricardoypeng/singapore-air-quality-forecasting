# Reproducible environment for the whole pipeline.
#
# Build:   docker build -t sg-pm25 .
# Train:   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/models:/app/models" sg-pm25
# Predict: docker run --rm -v "$PWD/models:/app/models" sg-pm25 python -m src.predict

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/singapore-pm25-hourly-raw.csv data/

CMD ["python", "-m", "src.train"]
