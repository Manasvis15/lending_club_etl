import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def load_to_postgres(df: pd.DataFrame, table: str = "lending_club_loans", chunksize: int = 5000) -> None:
    """Load DataFrame into Postgres in chunks."""
    engine = get_engine()
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    print(f"[Load] {len(df):,} rows → {table}")

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/transformed.parquet")
    load_to_postgres(df)