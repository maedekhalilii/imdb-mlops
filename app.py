from fastapi import FastAPI
import joblib
import re

app = FastAPI()

# load model

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


@app.get("/")
def home():
    return {"message": "IMDb Sentiment API"}

@app.post("/predict_batch")
def predict_batch(reviews: list[str]):
    cleaned_reviews = [clean_text(r) for r in reviews]
    vectors = vectorizer.transform(cleaned_reviews)
    predictions = model.predict(vectors)
    results = ["Positive" if p == 1 else "Negative" for p in predictions]
    return {"predictions": results}
