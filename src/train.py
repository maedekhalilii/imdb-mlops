import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocessing import preprocess_dataframe

def train_model(
    data_path: str,
    model_output_dir: str,
    max_features: int = 20000,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Loads dataset, preprocesses, vectorizes and trains a Logistic Regression model.
    Saves model + vectorizer inside model_output_dir.
    """

    df = pd.read_csv(data_path)
    df = preprocess_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"],
        test_size=test_size,
        random_state=random_state
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    os.makedirs(model_output_dir, exist_ok=True)

    joblib.dump(model, f"{model_output_dir}/sentiment_model.pkl")
    joblib.dump(vectorizer, f"{model_output_dir}/tfidf_vectorizer.pkl")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
