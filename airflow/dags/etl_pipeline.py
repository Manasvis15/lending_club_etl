import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/hp/projects/lending_club_fin")
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_RAW     = PROJECT_DIR / "data" / "raw" / "data_80.csv"
DATA_PROC    = PROJECT_DIR / "data" / "processed"

sys.path.insert(0, str(SCRIPTS_DIR))

# ── Task callables ─────────────────────────────────────────────────────────────

def run_ingestion(**context):
    from ingestion import load_raw_data, save_raw_snapshot
    df = load_raw_data(str(DATA_RAW))
    save_raw_snapshot(df, str(DATA_PROC / "raw_snapshot.parquet"))
    print(f"[DAG] Ingestion complete — {len(df):,} rows")
    

def run_cleaning(**context):
    import pandas as pd
    from cleaning import clean_pipeline

    snapshot = DATA_PROC / "raw_snapshot.parquet"
    df = pd.read_parquet(snapshot)
    df_clean = clean_pipeline(df)
    out = DATA_PROC / "cleaned.parquet"
    df_clean.to_parquet(out, index=False)
    print(f"[DAG] Cleaning complete — {df_clean.shape}")


def run_transformation(**context):
    import pandas as pd
    from transformation import transform_pipeline

    cleaned = DATA_PROC / "cleaned.parquet"
    df = pd.read_parquet(cleaned)
    transform_pipeline(df)          # saves transformed.parquet internally
    print("[DAG] Transformation complete")


def run_load(**context):
    import pandas as pd
    import psycopg2
    from psycopg2.extras import execute_values
    from dotenv import load_dotenv

    load_dotenv(str(PROJECT_DIR / ".env"))

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cur = conn.cursor()

    transformed = DATA_PROC / "transformed.parquet"
    df = pd.read_parquet(transformed)

    # Replace NaN with None for postgres compatibility
    df = df.where(pd.notnull(df), None)

    # Drop and recreate table
    cur.execute("DROP TABLE IF EXISTS lending_club_loans;")

    # Build CREATE TABLE from dataframe columns
    col_defs = ", ".join([f'"{col}" TEXT' for col in df.columns])
    cur.execute(f"CREATE TABLE lending_club_loans ({col_defs});")

    # Insert in chunks
    cols = ", ".join([f'"{c}"' for c in df.columns])
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        rows = [tuple(row) for row in chunk.itertuples(index=False)]
        execute_values(
            cur,
            f"INSERT INTO lending_club_loans ({cols}) VALUES %s",
            rows
        )
        print(f"[DAG] Inserted rows {i} to {i+len(chunk)}")

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DAG] Load complete — {len(df):,} rows → lending_club_loans")

# ── DAG definition ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "hp",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="lending_club_etl",
    default_args=default_args,
    description="Lending Club ETL: ingest → clean → transform → load",
    schedule="@daily",
    catchup=False,
    tags=["lending_club", "etl"],
) as dag:

    ingest = PythonOperator(
        task_id="ingestion",
        python_callable=run_ingestion,
    )

    clean = PythonOperator(
        task_id="cleaning",
        python_callable=run_cleaning,
    )

    transform = PythonOperator(
        task_id="transformation",
        python_callable=run_transformation,
    )

    load = PythonOperator(
        task_id="load_to_postgres",
        python_callable=run_load,
    )

    ingest >> clean >> transform >> load
