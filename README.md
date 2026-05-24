# IMDb Sentiment Analysis MLOps Pipeline

This project is part of a university MLOps course focused on building an end-to-end machine learning workflow for sentiment analysis using the IMDb movie reviews dataset.

---

# Project Goals

The goal of this project is not only to train a machine learning model, but also to explore key MLOps concepts such as:

- Reproducibility
- Data Versioning
- Workflow Orchestration
- Containerization
- API Deployment
- Team Collaboration

---

# Technologies Used

- Git & GitHub
- DVC
- Apache Airflow
- FastAPI
- Docker
- Scikit-learn
- Pandas
- NLTK

---

# Project Architecture

IMDb Dataset
↓
DVC Tracking
↓
Airflow Pipeline
↓
Preprocessing
↓
Feature Engineering
↓
Model Training
↓
FastAPI API
↓
Docker Container

---

# Project Structure

```text
imdb-mlops/
│
├── airflow/
│   └── dags/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── src/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
```

# DVC Usage

Initialize DVC:

```bash
dvc init
```

Track dataset:

```bash
dvc add data/raw/IMDB-Dataset.csv
```

# Docker Usage

Build image:

```bash
docker build -t imdb-sentiment .
```

Run container:

```bash
docker run -p 8000:8000 imdb-sentiment
```

# API Usage

Run API locally:

```bash
uvicorn app:app --reload
```

Swagger UI:

```text
http://localhost:8000/docs
```

---

Developed as part of the Special Topics MLOps course project.