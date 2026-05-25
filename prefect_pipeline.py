from prefect import flow, task
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@task
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    return df


@task
def clean_data(df):
    df["review"] = df["review"].apply(
        lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower())
    )
    return df


@task
def train_model(df):

    X = df["review"]
    y = df["sentiment"]

    vectorizer = TfidfVectorizer(max_features=5000)

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Accuracy: {acc}")

    joblib.dump(model, "sentiment_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


@flow
def imdb_sentiment_flow():

    df = load_data()

    df = clean_data(df)

    train_model(df)


if __name__ == "__main__":
    imdb_sentiment_flow()