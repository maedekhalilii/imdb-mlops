import os
import yaml
import pandas as pd
from prefect import task, flow
import mlflow
import mlflow.sklearn

from src.preprocessing import preprocess_dataframe, tokenize_dataframe
from src.train import train_model

# -------------------------------------------------------
# Load configuration
# -------------------------------------------------------
def load_config():
    config_path = os.path.join("config", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()


# -------------------------------------------------------
# Prefect Tasks
# -------------------------------------------------------
@task
def load_data():
    df = pd.read_csv(config["data"]["raw_path"])
    return df

@task
def clean_data(df: pd.DataFrame):
    return preprocess_dataframe(df)

@task
def tokenize(df: pd.DataFrame):
    return tokenize_dataframe(df)

@task
def validate(df: pd.DataFrame):
    required = ["review", "sentiment"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return True

@task
def train(df: pd.DataFrame):
    mlflow.set_experiment("IMDB_sentiment_analysis")

    with mlflow.start_run():
        results = train_model(
            data_path=config["data"]["raw_path"],
            model_output_dir=config["model"]["output_dir"],
            max_features=config["params"]["max_features"],
            test_size=config["params"]["test_size"],
            random_state=config["params"]["random_state"],
        )

        mlflow.log_params(config["params"])
        mlflow.log_metrics(results)

    return results


# -------------------------------------------------------
# Prefect Flow
# -------------------------------------------------------
@flow(name="IMDB Sentiment Pipeline")
def imdb_pipeline():
    df = load_data()
    df = clean_data(df)
    df = tokenize(df)
    validate(df)
    metrics = train(df)
    return metrics


# -------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------
if __name__ == "__main__":
    imdb_pipeline()
    print("Pipeline finished successfully.")
