import joblib
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app import app

client = TestClient(app)


# ── health & home ──────────────────────────────────────────────────────────────

def test_home():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "running"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["vectorizer_loaded"] is True


# ── /predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_200():
    r = client.post("/predict", json={"review": "This movie was great!"})
    assert r.status_code == 200


def test_predict_schema():
    r = client.post("/predict", json={"review": "Excellent film."})
    data = r.json()
    assert "prediction_id" in data
    assert "sentiment" in data
    assert data["sentiment"] in ("positive", "negative")
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], float)


def test_predict_invalid_input_returns_422():
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_predict_empty_review():
    r = client.post("/predict", json={"review": ""})
    assert r.status_code == 200
    assert r.json()["sentiment"] in ("positive", "negative")


# ── /predict_batch ─────────────────────────────────────────────────────────────

def test_predict_batch_returns_200():
    r = client.post("/predict_batch", json={"reviews": ["Good movie", "Bad movie"]})
    assert r.status_code == 200


def test_predict_batch_schema():
    r = client.post("/predict_batch", json={"reviews": ["Great!", "Terrible!"]})
    data = r.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2
    assert all(s in ("positive", "negative") for s in data["results"])


def test_predict_batch_invalid_input_returns_422():
    r = client.post("/predict_batch", json={})
    assert r.status_code == 422


# ── /feedback ──────────────────────────────────────────────────────────────────

def test_feedback_saved():
    pred = client.post("/predict", json={"review": "I loved it!"})
    pid = pred.json()["prediction_id"]

    r = client.post("/feedback", json={"prediction_id": pid, "correct_label": "positive"})
    assert r.status_code == 200
    assert r.json()["status"] == "saved"


def test_feedback_invalid_label_returns_422():
    r = client.post("/feedback", json={"prediction_id": "abc", "correct_label": "unknown"})
    assert r.status_code == 422


# ── /metrics ───────────────────────────────────────────────────────────────────

def test_metrics_latency_returns_200():
    r = client.get("/metrics/latency")
    assert r.status_code == 200
    data = r.json()
    assert "mean_latency_ms" in data
    assert "total_requests" in data


def test_metrics_accuracy_returns_200():
    r = client.get("/metrics/accuracy")
    assert r.status_code == 200


# ── model files ────────────────────────────────────────────────────────────────

def test_model_file_exists():
    assert Path("models/sentiment_model_quantized.pkl").exists()


def test_vectorizer_file_exists():
    assert Path("models/tfidf_vectorizer.pkl").exists()


def test_model_loads():
    m = joblib.load("models/sentiment_model_quantized.pkl")
    assert m is not None


def test_vectorizer_has_vocabulary():
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    assert hasattr(vec, "vocabulary_")
    assert len(vec.vocabulary_) > 0
