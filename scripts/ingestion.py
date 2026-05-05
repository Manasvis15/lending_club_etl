import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
    # 1. Get the directory where ingestion.py is located
SCRIPT_DIR = Path(__file__).resolve().parent

# 3. Load the data
# try:
#     df = pd.read_csv(DATA_PATH)
#     print("Success! Data loaded.")
# except FileNotFoundError:
#     print(f"Still can't find it. Looked at: {DATA_PATH}")

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw Lending Club CSV and return DataFrame."""

# 2. Go up one level to 'project/', then down into 'data/raw/'
    filepath = SCRIPT_DIR.parent / "data" / "raw" / "data_80.csv"

    df = pd.read_csv(filepath, low_memory=False, dtype=str)
    print(f"[Ingestion] Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df

def save_raw_snapshot(df: pd.DataFrame, out_path: str) -> None:
    """Save a parquet snapshot of raw data."""
    out_path = SCRIPT_DIR.parent / "data" / "processed" / "raw_snapshot.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[Ingestion] Snapshot saved → {out_path}")

if __name__ == "__main__":
    df = load_raw_data("data/raw/data_80.csv")
    save_raw_snapshot(df, "data/processed/raw_snapshot.parquet")