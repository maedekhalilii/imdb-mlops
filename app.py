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


@app.post("/predict")
def predict(review: str):

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "review": review,
        "sentiment": sentiment
    }