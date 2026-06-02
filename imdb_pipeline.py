import os
import re
import yaml
import joblib
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from prefect import flow, task

# --- تابعی برای بارگذاری تنظیمات از پوشه config ---
def load_config():
    # پیدا کردن مسیر مطلق فایل فعلی برای جلوگیری از خطای مسیر در محیط‌های مختلف
    base_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_path, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# بارگذاری کانفیگ در سطح Global برای استفاده در تسک‌ها
config = load_config()

@task
def install_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')

@task
def load_and_clean():
    # استفاده از مسیر دیتاست از فایل کانفیگ
    df = pd.read_csv(config['data']['raw_path'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    df["review"] = df["review"].apply(clean_text)
    print("Data cleaning completed. Sample:")
    print(df.head(2))
    return df

@task
def tokenize_text(df):
    df["tokens"] = df["review"].apply(word_tokenize)
    return df

@task
def train_model(df):
    # استفاده از پارامترهای هایپرپارامتری از کانفیگ
    vectorizer = TfidfVectorizer(max_features=config['params']['max_features'])
    X = vectorizer.fit_transform(df["review"])
    
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["sentiment"],
        test_size=config['params']['test_size'],
        random_state=config['params']['random_state']
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Training Completed. Accuracy: {accuracy:.4f}")

    # --- مدیریت ذخیره‌سازی در پوشه Models طبق کانفیگ ---
    model_dir = config['paths']['model_dir']
    os.makedirs(model_dir, exist_ok=True) # ساخت پوشه اگر وجود نداشت

    model_save_path = os.path.join(model_dir, config['paths']['model_name'])
    vec_save_path = os.path.join(model_dir, config['paths']['vectorizer_name'])

    joblib.dump(model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)
    
    print(f"Artifacts saved successfully in '{model_dir}' folder.")
    return model, vectorizer

@task
def test_sentence(model, vectorizer):
    sentence = input("\nEnter a review for testing: ")
    sentence = sentence.lower()
    sentence = re.sub(r"<.*?>", "", sentence)
    sentence = re.sub(r"[^a-zA-Z\s]", "", sentence)

    vector = vectorizer.transform([sentence])
    prediction = model.predict(vector)[0]

    result = "Positive" if prediction == 1 else "Negative"
    print(f"Prediction Result: {result} Review")

@flow(name="imdb_sentiment_pipeline_v2")
def imdb_pipeline():
    install_nltk()
    df = load_and_clean()
    df = tokenize_text(df)
    model, vectorizer = train_model(df)
    test_sentence(model, vectorizer)

if __name__ == "__main__":
    imdb_pipeline()
