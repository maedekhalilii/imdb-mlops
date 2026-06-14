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
#from sklearn.metrics import accuracy_score
from prefect import flow, task
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

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
'''
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
'''

@task
def validate_data(df):
    """
    اعتبارسنجی کیفیت داده‌ها قبل از آموزش مدل
    بررسی: وجود ستون‌ها، مقادیر خالی، مقادیر نامعتبر، تعادل کلاس‌ها
    """
    print("\n=== Starting Data Validation ===")
    
    # 1. بررسی وجود ستون‌های مورد نیاز
    required_columns = ['review', 'sentiment']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"❌ ستون '{col}' در دیتاست وجود ندارد!")
    
    # 2. بررسی مقادیر خالی (Null)
    before_count = len(df)
    null_reviews = df['review'].isnull().sum()
    null_sentiments = df['sentiment'].isnull().sum()
    
    if null_reviews > 0 or null_sentiments > 0:
        print(f"⚠️ هشدار: {null_reviews} ردیف review خالی و {null_sentiments} ردیف sentiment خالی")
        df = df.dropna(subset=['review', 'sentiment'])
        print(f"   {before_count - len(df)} ردیف خالی حذف شد")
    
    # 3. بررسی مقدارهای معتبر برای Sentiment
    valid_sentiments = ['positive', 'negative']
    invalid_mask = ~df['sentiment'].isin(valid_sentiments)
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        print(f"⚠️ هشدار: {invalid_count} ردیف sentiment نامعتبر یافت شد")
        print(f"   مقادیر نامعتبر: {df[invalid_mask]['sentiment'].unique()}")
        df = df[~invalid_mask]
        print(f"   {invalid_count} ردیف نامعتبر حذف شد")
    
    # 4. بررسی خالی نبودن متن نظرات (بررسی رشته‌های خالی)
    empty_reviews = df[df['review'].str.strip() == '']
    empty_count = len(empty_reviews)
    
    if empty_count > 0:
        print(f"⚠️ هشدار: {empty_count} ردیف دارای متن خالی هستند")
        df = df[df['review'].str.strip() != '']
        print(f"   {empty_count} ردیف خالی حذف شد")
    
    # 5. بررسی تعادل کلاس‌ها (Class Balance)
    sentiment_counts = df['sentiment'].value_counts()
    print(f"\n📊 توزیع کلاس‌ها پس از پاکسازی:")
    print(f"   Positive: {sentiment_counts.get('positive', 0)}")
    print(f"   Negative: {sentiment_counts.get('negative', 0)}")
    
    if len(sentiment_counts) == 2:
        min_class = min(sentiment_counts)
        max_class = max(sentiment_counts)
        balance_ratio = min_class / max_class
        
        if balance_ratio < 0.3:
            print(f"⚠️ هشدار شدید: داده‌ها بسیار نامتوازن هستند! (نسبت: {balance_ratio:.2f})")
            print(f"   ممکن است مدل به سمت کلاس اکثریت bias داشته باشد.")
        elif balance_ratio < 0.7:
            print(f"⚠️ هشدار: داده‌ها نسبتاً نامتوازن هستند (نسبت: {balance_ratio:.2f})")
    
    # 6. گزارش نهایی
    final_count = len(df)
    removed_count = before_count - final_count
    print(f"\n✅ اعتبارسنجی تکمیل شد:")
    print(f"   تعداد اولیه: {before_count}")
    print(f"   تعداد حذف شده: {removed_count}")
    print(f"   تعداد نهایی: {final_count}")
    print("=== Data Validation Completed ===\n")
    
    return df

@task
def train_model(df):

    # شروع MLflow Run
    mlflow.start_run()

    # پارامترها
    max_features = config['params']['max_features']
    test_size = config['params']['test_size']
    random_state = config['params']['random_state']

    # ثبت پارامترها
    mlflow.log_param("max_features", max_features)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)

    X = vectorizer.fit_transform(df["review"])

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["sentiment"],
        test_size=test_size,
        random_state=random_state
    )

    # Model
    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    # ثبت Metrics در MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Confusion Matrix


    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d'
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    # ذخیره Artifact
    mlflow.log_artifact("confusion_matrix.png")

    plt.close()


    # ذخیره مدل‌ها
 

    model_dir = config['paths']['model_dir']

    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(
        model_dir,
        config['paths']['model_name']
    )

    vec_save_path = os.path.join(
        model_dir,
        config['paths']['vectorizer_name']
    )

    joblib.dump(model, model_save_path)
    joblib.dump(vectorizer, vec_save_path)

    # ثبت مدل در MLflow
    mlflow.sklearn.log_model(
        model,
        "sentiment_model"
    )

    print("Artifacts saved successfully")

    # end Run
    mlflow.end_run()

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
    df = validate_data(df) 
    model, vectorizer = train_model(df)
    test_sentence(model, vectorizer)

if __name__ == "__main__":
    imdb_pipeline()
