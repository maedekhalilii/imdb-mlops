from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

import pandas as pd
import re
import nltk
import joblib

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




def install_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')


def load_and_clean():
    global df
    DATA_PATH = "/opt/airflow/dags/IMDB Dataset.csv"
    df = pd.read_csv(DATA_PATH)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    df["review"] = df["review"].apply(clean_text)

    print(df.head())


def tokenize_text():
    global df

    df["tokens"] = df["review"].apply(word_tokenize)

    print(df["tokens"].head())


def train_model():
    global df

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

    
    joblib.dump(vectorizer, '/opt/airflow/models/tfidf_vectorizer.pkl')
    joblib.dump(model, '/opt/airflow/models/sentiment_model.pkl')
    print("Model Saved")




with DAG(
    dag_id='imdb_sentiment_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:

    task1 = PythonOperator(
        task_id='download_nltk',
        python_callable=install_nltk
    )

    task2 = PythonOperator(
        task_id='load_and_clean_data',
        python_callable=load_and_clean
    )

    task3 = PythonOperator(
        task_id='tokenize_reviews',
        python_callable=tokenize_text
    )

    task4 = PythonOperator(
        task_id='train_sentiment_model',
        python_callable=train_model
    )

    
    task1 >> task2 >> task3 >> task4