import argparse
import pickle
import os
import math
from pathlib import Path
import joblib

import pandas as pd
import mlflow
import mlflow.sklearn

from sqlalchemy import create_engine
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def train_and_log(user, password, host, port, db_name, table_name):
    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "nyc-taxi-experiment"

    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(name=experiment_name)

    mlflow.set_experiment(experiment_name)

    print("Reading data...")
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

    print("Preparing data...")
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    y = df['duration'].values

    split_idx = int(X.shape[0] * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("Training model...")
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("num_features", X.shape[1])
        print(f"Model intercept: {lr.intercept_}")

        # Логируем модель через mlflow
        mlflow.sklearn.log_model(lr, artifact_path="model")
        print("Model logged to MLflow")

        # Логируем DictVectorizer как артефакт
        dv_path = "dv.pkl"
        with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(dv_path, artifact_path="preprocessor")
        print("DictVectorizer saved and logged as artifact")

        # Оценка размера модели (локальный файл из артефактов)
        local_model_path = Path("mlruns") / run.info.experiment_id / run.info.run_id / "artifacts" / "model" / "model.pkl"
        if local_model_path.exists():
            model_size_bytes = os.path.getsize(local_model_path)
            mlflow.log_metric("model_size_bytes", model_size_bytes)
            print(f"Model size: {model_size_bytes} bytes")
        else:
            print("Warning: model.pkl not found to measure size.")

        print(f"Validation RMSE: {rmse:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--db_name", required=True)
    parser.add_argument("--table_name", required=True)
    args = parser.parse_args()

    train_and_log(**vars(args))
