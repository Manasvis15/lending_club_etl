from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.insert(0, "/home/hp/projects/lending_club_fin/scripts")

from ingestion import load_raw_data, save_raw_snapshot
from cleaning import clean_pipeline
from transformation import transform_pipeline
from load_to_postgres import load_to_postgres, load_predictions_to_postgres
from default_prediction import (load_data, prepare_features, split_data,
                                 train_logistic_regression, train_random_forest,
                                 train_xgboost, save_model)
from risk_segmentation import (prepare_clustering_features, plot_elbow,
                                plot_silhouette, fit_kmeans, save_clustering_model)
import pandas as pd

BASE_DIR = "/home/hp/projects/lending_club_fin"


def ingest():
    df = load_raw_data(f"{BASE_DIR}/data/raw/data_80.csv")
    save_raw_snapshot(df, f"{BASE_DIR}/data/processed/raw_snapshot.parquet")


def clean():
    df = pd.read_parquet(f"{BASE_DIR}/data/processed/raw_snapshot.parquet")
    cleaned = clean_pipeline(df)
    cleaned.to_parquet(f"{BASE_DIR}/data/processed/cleaned.parquet", index=False)


def transform():
    df = pd.read_parquet(f"{BASE_DIR}/data/processed/cleaned.parquet")
    transform_pipeline(df)


def load_loans():
    df = pd.read_parquet(f"{BASE_DIR}/data/processed/transformed.parquet")
    load_to_postgres(df)


def train_models():
    df = load_data()

    # Default prediction
    X, y, ids = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    lr_model, scaler, _, _ = train_logistic_regression(X_train, X_test, y_train, y_test)
    _, _, _ = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, _, _ = train_xgboost(X_train, X_test, y_train, y_test)
    save_model(xgb_model, scaler, X.columns)

    # Risk segmentation — use sample + fixed K
    df_sample = df.sample(frac=0.1, random_state=42)
    X_cluster, X_scaled, cluster_cols, cluster_scaler = prepare_clustering_features(df_sample.copy())
    km_model, _ = fit_kmeans(X_scaled, k=4)
    save_clustering_model(km_model, cluster_scaler, cluster_cols)

def load_predictions():
    """Load predictions to postgres after models are trained."""
    df = pd.read_parquet(f"{BASE_DIR}/data/processed/transformed.parquet")
    load_predictions_to_postgres(df)


with DAG(
    dag_id="lending_club_etl",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["lending-club", "etl"],
) as dag:

    t1 = PythonOperator(task_id="ingest",            python_callable=ingest)
    t2 = PythonOperator(task_id="clean",             python_callable=clean)
    t3 = PythonOperator(task_id="transform",         python_callable=transform)
    t4 = PythonOperator(task_id="load_loans",        python_callable=load_loans)
    t5 = PythonOperator(task_id="train_models",      python_callable=train_models)
    t6 = PythonOperator(task_id="load_predictions",  python_callable=load_predictions)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6