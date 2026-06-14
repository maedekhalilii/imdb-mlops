# IMDb Sentiment Analysis MLOps Pipeline

This project implements a comprehensive MLOps workflow for sentiment analysis on IMDb movie reviews, covering everything from data versioning to automated CI/CD and monitoring.

---

##  Project Goals
- **End-to-End Automation:** Automating the ML lifecycle from data ingestion to deployment.
- **Model Monitoring:** Tracking performance and detecting data/model drift.
- **Robust CI/CD:** Ensuring code quality and preventing broken builds using GitHub Actions.
- **Containerization:** Standardizing the environment for seamless deployment.

---

##  Technologies Used
- **Orchestration:** Prefect & Apache Airflow
- **Tracking & Monitoring:** MLflow
- **Data Versioning:** DVC
- **API:** FastAPI
- **Containerization:** Docker & Docker Compose
- **CI/CD:** GitHub Actions (Linting & Pytest)

---

## Project Structure
Based on the current repository architecture:

```text
imdb-mlops/
│
├── .github/workflows/    # CI/CD Pipeline (Linting, Tests)
├── .dvc/                 # DVC configuration
├── config/               # Configuration files (YAML/JSON)
├── data/raw/             # Raw dataset (Versioned by DVC)
├── drift/                # Scripts for Model/Data Drift detection
├── logs/                 # Application and Pipeline logs
├── models/               # Serialized model files (.pkl)
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── reports/              # Performance and drift reports
├── src/                  # Core source code (Preprocessing, Training)
├── tests/                # Automated unit tests
├── app.py                # FastAPI deployment script
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

---

##  Key Workflows

### 1. Data & Model Pipeline
We use **Prefect** and **Airflow** to orchestrate the ML pipeline.
- `prefect_pipeline.py`: Handles workflow tasks.
- `imdb_pipeline.py`: Main execution logic for training.

### 2. CI/CD & Branch Protection
The project is protected by GitHub Rulesets.
- **Continuous Integration:** Every Push/PR triggers `lint-and-test` via GitHub Actions.
- **Status Checks:** Merging to `main` is blocked unless all tests pass.

### 3. Deployment (Docker)
To run the entire stack (API + Monitoring):
```bash
docker-compose up --build
```
The API will be available at `port 8000`.

---

## 📊 Monitoring & Optimization
- **Drift Detection:** Located in the `/drift` folder to ensure model reliability over time.
- **Optimization:** `benchmark_optimization.py` is used for performance tuning and latency checks.

