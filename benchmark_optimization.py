import os
import time
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/sentiment_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"
DATA_PATH = "data/raw/IMDB Dataset.csv"
QUANT_MODEL_PATH = "models/sentiment_model_quantized.pkl"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


print("Loading Model and Data...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
df = pd.read_csv(DATA_PATH).sample(500, random_state=42)
df['review'] = df['review'].apply(clean_text)

texts = df['review'].tolist()
actuals = df['sentiment'].tolist()  # keep as 'positive'/'negative' to match model output

# --- 1. Baseline (single inference, one-by-one) ---
print("\n[1] Running Baseline (Single Inference)...")
start_time = time.time()
baseline_preds = []
for text in texts:
    vec = vectorizer.transform([text])
    baseline_preds.append(model.predict(vec)[0])
baseline_duration = time.time() - start_time
baseline_acc = accuracy_score(actuals, baseline_preds)
baseline_size = os.path.getsize(MODEL_PATH) / 1024

# --- 2. Batching (vectorize all at once) ---
print("[2] Running Batching (full batch)...")
start_time = time.time()
X_batch = vectorizer.transform(texts)
batch_preds = model.predict(X_batch)
batch_duration = time.time() - start_time
batch_acc = accuracy_score(actuals, batch_preds)

# --- 3. Quantization (Float64 -> Float32) ---
print("[3] Applying Quantization (Float64 -> Float32)...")
model.coef_ = model.coef_.astype(np.float32)
model.intercept_ = model.intercept_.astype(np.float32)
joblib.dump(model, QUANT_MODEL_PATH)

start_time = time.time()
X_quant = vectorizer.transform(texts)
quant_preds = model.predict(X_quant)
quant_duration = time.time() - start_time
quant_acc = accuracy_score(actuals, quant_preds)
quant_size = os.path.getsize(QUANT_MODEL_PATH) / 1024

# --- Final Report ---
print("\n" + "=" * 40)
print("         BENCHMARK REPORT")
print("=" * 40)
results = {
    "Metric": ["Inference Time (s)", "Accuracy", "Model Size (KB)"],
    "Baseline": [
        f"{baseline_duration:.4f}",
        f"{baseline_acc:.4f}",
        f"{baseline_size:.2f}"
    ],
    "Batching": [
        f"{batch_duration:.4f}",
        f"{batch_acc:.4f}",
        f"{baseline_size:.2f}"
    ],
    "Quantized": [
        f"{quant_duration:.4f}",
        f"{quant_acc:.4f}",
        f"{quant_size:.2f}"
    ],
}
report_df = pd.DataFrame(results)
print(report_df.to_string(index=False))
print("\nQuantized model saved to:", QUANT_MODEL_PATH)
