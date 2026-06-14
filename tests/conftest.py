"""
Creates minimal stub models if the real pkl files are missing (e.g. in CI).
Runs automatically before any test via pytest_configure.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def pytest_configure(config):
    model_path = Path("models/sentiment_model_quantized.pkl")
    vec_path = Path("models/tfidf_vectorizer.pkl")

    if not model_path.exists() or not vec_path.exists():
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        texts = [
            "great movie loved it excellent",
            "terrible film hated it awful",
            "wonderful acting brilliant story",
            "boring waste of time bad",
        ]
        labels = ["positive", "negative", "positive", "negative"]

        vec = TfidfVectorizer()
        X = vec.fit_transform(texts)
        clf = LogisticRegression()
        clf.fit(X, labels)

        joblib.dump(clf, str(model_path))
        joblib.dump(vec, str(vec_path))
        print("\n[conftest] Stub models created for CI testing.")

    Path("logs").mkdir(exist_ok=True)
