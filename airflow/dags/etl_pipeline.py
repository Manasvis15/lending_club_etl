from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.insert(0, "/opt/airflow/scripts")  # adjust path as needed

from ingestion import load_raw_data, save_raw_snapshot
from cleaning import clean_pipeline
from transformation import transform_pipeline
from load_to_postgres import load_to_postgres
import pandas as pd

RAW_PATH = "data/raw/accepted_2007_to_2018Q4.csv"

def ingest():
    df = load_raw_data(RAW_PATH)
    save_raw_snapshot(df, "data/processed/raw_snapshot.parquet")

def clean():
    df = pd.read_parquet("data/processed/raw_snapshot.parquet")
    cleaned = clean_pipeline(df)
    cleaned.to_parquet("data/processed/cleaned.parquet", index=False)

def transform():
    df = pd.read_parquet("data/processed/cleaned.parquet")
    transform_pipeline(df)

def load():
    df = pd.read_parquet("data/processed/transformed.parquet")
    load_to_postgres(df)

with DAG(
    dag_id="lending_club_etl",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@once",
    catchup=False,
    tags=["lending-club", "etl"],
) as dag:

    t1 = PythonOperator(task_id="ingest",    python_callable=ingest)
    t2 = PythonOperator(task_id="clean",     python_callable=clean)
    t3 = PythonOperator(task_id="transform", python_callable=transform)
    t4 = PythonOperator(task_id="load",      python_callable=load)

    t1 >> t2 >> t3 >> t4