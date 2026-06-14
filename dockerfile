FROM airflow_imdb-pipeline:latest

WORKDIR /app

# Copy updated source code and models on top of existing image
COPY . /app

# Ensure required directories exist
RUN mkdir -p /app/models /app/data/raw /app/data/processed /app/reports /app/mlflow_runs

# Ports: 8000=FastAPI, 5000=MLflow
EXPOSE 8000
EXPOSE 5000

ENV MLFLOW_TRACKING_URI=sqlite:///app/mlflow_runs/mlflow.db
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the FastAPI model server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
