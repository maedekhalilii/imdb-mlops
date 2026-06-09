import os
import time
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score

# --- تنظیمات و بارگذاری ---
MODEL_PATH = "models/sentiment_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"
DATA_PATH = "data/raw/IMDB Dataset.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# بارگذاری مدل و دیتا برای تست
print("Loading Model and Data...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
df = pd.read_csv(DATA_PATH).sample(500) # تست روی 500 نمونه برای سرعت
df['review'] = df['review'].apply(clean_text)
df['sentiment'] = df['sentiment'].map({"positive": 1, "negative": 0})

texts = df['review'].tolist()
actuals = df['sentiment'].tolist()

# --- 1. بنچ‌مارک Baseline (تک‌تک) ---
print("\n[1] Running Baseline (Single Inference)...")
start_time = time.time()
baseline_preds = []
for text in texts:
    vec = vectorizer.transform([text])
    baseline_preds.append(model.predict(vec)[0])
baseline_duration = time.time() - start_time
baseline_acc = accuracy_score(actuals, baseline_preds)

# --- 2. بنچ‌مارک Batching (پردازش دسته‌ای) ---
print("[2] Running Batching (Batch Size: 100)...")
start_time = time.time()
# در Batching کل لیست را یکباره به وکتورایزر می‌دهیم
X_batch = vectorizer.transform(texts)
batch_preds = model.predict(X_batch)
batch_duration = time.time() - start_time

# --- 3. پیاده‌سازی Quantization (کاهش دقت وزن‌ها) ---
print("[3] Applying Quantization (Float64 -> Float32)...")
# تغییر نوع داده‌های وزن مدل از float64 به float32 برای کاهش حجم و افزایش سرعت CPU
model.coef_ = model.coef_.astype(np.float32)
model.intercept_ = model.intercept_.astype(np.float32)

# ذخیره مدل کوانتایز شده برای چک کردن حجم
QUANT_MODEL_PATH = "models/sentiment_model_quantized.pkl"
joblib.dump(model, QUANT_MODEL_PATH)

# تست مدل کوانتایز شده
start_time = time.time()
X_quant = vectorizer.transform(texts)
quant_preds = model.predict(X_quant)
quant_duration = time.time() - start_time
quant_acc = accuracy_score(actuals, quant_preds)

# --- گزارش نهایی ---
print("\n" + "="*30)
print("   FINAL BENCHMARK REPORT")
print("="*30)
results = {
    "Metric": ["Inference Time (s)", "Accuracy", "Model Size (KB)"],
    "Baseline": [f"{baseline_duration:.4f}", f"{baseline_acc:.4f}", f"{os.path.getsize(MODEL_PATH)/1024:.2f}"],
    "Batching": [f"{batch_duration:.4f}", f"{baseline_acc:.4f}", f"{os.path.getsize(MODEL_PATH)/1024:.2f}"],
    "Quantized": [f"{quant_duration:.4f}", f"{quant_acc:.4f}", f"{os.path.getsize(QUANT_MODEL_PATH)/1024:.2f}"]
}
report_df = pd.DataFrame(results)
print(report_df)
