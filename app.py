import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


# Load Model & Vectorizer

MODEL_PATH = "models/sentiment_model_quantized.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)


# FastAPI App

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Sentiment Analysis using TF-IDF + Logistic Regression",
    version="1.0"
)


# Request Schemas

class ReviewRequest(BaseModel):
    review: str


class BatchRequest(BaseModel):
    reviews: list[str]


# Text Cleaning

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# Home Endpoint

@app.get("/")
def home():
    return {
        "project": "IMDb Sentiment Analysis",
        "status": "running"
    }


# Health Check

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "vectorizer_loaded": True
    }


# Single Prediction

@app.post("/predict")
def predict(data: ReviewRequest):

    review = clean_text(data.review)

    vector = vectorizer.transform([review])

    prediction = model.predict(vector)[0]

    sentiment = (
        "positive"
        if prediction == 1
        else "negative"
    )

    return {
        "review": data.review,
        "sentiment": sentiment
    }


# Batch Prediction

@app.post("/predict_batch")
def predict_batch(data: BatchRequest):

    cleaned_reviews = [
        clean_text(review)
        for review in data.reviews
    ]

    vectors = vectorizer.transform(
        cleaned_reviews
    )

    predictions = model.predict(
        vectors
    )

    results = [
        "positive"
        if pred == 1
        else "negative"
        for pred in predictions
    ]

    return {
        "count": len(results),
        "results": results
    }