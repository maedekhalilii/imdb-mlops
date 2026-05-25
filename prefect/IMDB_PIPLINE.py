from prefect import flow, task
import pandas as pd
import re
import nltk
import joblib

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




@task
def install_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')


@task
def load_and_clean():
    
    df = pd.read_csv("IMDB Dataset.csv")

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    df["review"] = df["review"].apply(clean_text)

    print(df.head())

    return df


@task
def tokenize_text(df):

    df["tokens"] = df["review"].apply(word_tokenize)

    print(df["tokens"].head())

    return df


@task
def train_model(df):

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(df["review"])

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["sentiment"],
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(model, "sentiment_model.pkl")

    print("Model Saved")

    return model, vectorizer


@task
def test_sentence(model, vectorizer):

    sentence = input("Enter a review: ")

    sentence = sentence.lower()
    sentence = re.sub(r"<.*?>", "", sentence)
    sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)

    vector = vectorizer.transform([sentence])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        print("Positive Review")
    else:
        print("Negative Review")




@flow(name="imdb_sentiment_pipeline")
def imdb_pipeline():

    install_nltk()

    df = load_and_clean()

    df = tokenize_text(df)

    model, vectorizer = train_model(df)

    test_sentence(model, vectorizer)




if __name__ == "__main__":
    imdb_pipeline()