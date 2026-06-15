import re
import uuid
import time
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/sentiment_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"
PREDICTIONS_LOG = "logs/predictions.jsonl"
FEEDBACK_LOG = "logs/feedback.jsonl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
vocab = set(vectorizer.vocabulary_.keys())

Path("logs").mkdir(exist_ok=True)

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Sentiment Analysis using TF-IDF + Logistic Regression",
    version="2.0"
)


class ReviewRequest(BaseModel):
    review: str


class BatchRequest(BaseModel):
    reviews: list[str]


class FeedbackRequest(BaseModel):
    prediction_id: str
    correct_label: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def extract_features(text: str) -> dict:
    words = text.split()
    unknown = sum(1 for w in words if w not in vocab)
    return {
        "text_length": len(text),
        "word_count": len(words),
        "unknown_ratio": round(unknown / len(words), 4) if words else 0.0,
    }


def parse_prediction(raw_pred) -> str:
    if raw_pred in ("positive", "negative"):
        return str(raw_pred)
    return "positive" if int(raw_pred) == 1 else "negative"


def append_jsonl(path: str, entry: dict):
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/")
def home():
    return {"project": "IMDb Sentiment Analysis", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "vectorizer_loaded": True}


@app.post("/predict")
def predict(data: ReviewRequest):
    start = time.time()
    prediction_id = str(uuid.uuid4())

    cleaned = clean_text(data.review)
    features = extract_features(cleaned)
    vector = vectorizer.transform([cleaned])
    try:
        sentiment = parse_prediction(model.predict(vector)[0])
    except Exception:
        sentiment = "negative"

    latency_ms = round((time.time() - start) * 1000, 2)

    append_jsonl(PREDICTIONS_LOG, {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "review": data.review,
        **features,
        "sentiment": sentiment,
        "latency_ms": latency_ms,
    })

    return {
        "prediction_id": prediction_id,
        "review": data.review,
        "sentiment": sentiment,
        "latency_ms": latency_ms,
    }


@app.post("/predict_batch")
def predict_batch(data: BatchRequest):
    start = time.time()
    cleaned = [clean_text(r) for r in data.reviews]
    vectors = vectorizer.transform(cleaned)
    try:
        results = [parse_prediction(p) for p in model.predict(vectors)]
    except Exception:
        results = ["negative"] * len(data.reviews)

    latency_ms = round((time.time() - start) * 1000, 2)
    return {"count": len(results), "results": results, "latency_ms": latency_ms}


@app.post("/feedback")
def feedback(data: FeedbackRequest):
    if data.correct_label not in ("positive", "negative"):
        raise HTTPException(status_code=422, detail="correct_label must be 'positive' or 'negative'")
    append_jsonl(FEEDBACK_LOG, {
        "prediction_id": data.prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correct_label": data.correct_label,
    })
    return {"status": "saved", "prediction_id": data.prediction_id}


@app.get("/metrics/accuracy")
def metrics_accuracy():
    if not Path(PREDICTIONS_LOG).exists() or not Path(FEEDBACK_LOG).exists():
        return {"accuracy": None, "total_feedback": 0, "message": "not enough data"}

    preds = {}
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            e = json.loads(line)
            preds[e["prediction_id"]] = e["sentiment"]

    feedbacks = []
    with open(FEEDBACK_LOG) as f:
        for line in f:
            feedbacks.append(json.loads(line))

    if not feedbacks:
        return {"accuracy": None, "total_feedback": 0}

    correct = sum(1 for fb in feedbacks if preds.get(fb["prediction_id"]) == fb["correct_label"])
    return {
        "accuracy": round(correct / len(feedbacks), 4),
        "total_feedback": len(feedbacks),
        "correct": correct,
    }


@app.get("/metrics/latency")
def metrics_latency():
    if not Path(PREDICTIONS_LOG).exists():
        return {"mean_latency_ms": None, "p95_latency_ms": None, "total_requests": 0}

    latencies = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            e = json.loads(line)
            if "latency_ms" in e:
                latencies.append(e["latency_ms"])

    if not latencies:
        return {"mean_latency_ms": None, "p95_latency_ms": None, "total_requests": 0}

    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]
    return {
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": p95,
        "total_requests": len(latencies),
    }
