# MLOps Practice: Sentiment Analysis on IMDB Dataset

This project is part of our MLOps course, focusing on the end-to-end lifecycle of an NLP model—from data handling and pipeline orchestration to reproducibility and deployment.

## Project Structure
- `data/`: Contains raw and processed data (managed by DVC).
- `src/`: Core Python scripts for data cleaning, preprocessing, and model training.
- `notebooks/`: Exploratory Data Analysis (EDA) and initial model experiments.
- `airflow/`: DAGs and configuration files for pipeline orchestration.

## Key Technologies
- **Git**: Version control for code.
- **DVC**: Version control for large datasets.
- **Airflow**: Pipeline orchestration.
- **Python (Scikit-Learn/Pandas)**: NLP processing and model building.

## Reproducibility
To ensure experiments are reproducible:
1. Initialize DVC: `dvc init`
2. Pull data: `dvc pull`
3. Run the pipeline: (Instructions to be updated)

---
*Developed as part of the MLOps semester project.*
