FROM python:3.11

WORKDIR /app

# کپی فایل requirements.txt اول (برای کش بهتر)
COPY requirements.txt .

# نصب کتابخانه‌ها با timeout بیشتر
RUN pip install --default-timeout=100 -r requirements.txt

# دانلود داده‌های NLTK (اجباری برای tokenization)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# کپی بقیه فایل‌ها
COPY . /app

# ایجاد پوشه‌های مورد نیاز
RUN mkdir -p /app/models /app/mlflow_runs /app/data/processed /app/reports

# پورت‌های مورد نیاز در هفته دوم
EXPOSE 8000
EXPOSE 5000
EXPOSE 8157

# متغیرهای محیطی
ENV MLFLOW_TRACKING_URI=/app/mlflow_runs
ENV PYTHONUNBUFFERED=1

# دستور پیش‌فرض: اجرای پایپ‌لاین و سپس FastAPI
CMD ["sh", "-c", "python src/imdb_pipeline.py && uvicorn app:app --host 0.0.0.0 --port 8000"]