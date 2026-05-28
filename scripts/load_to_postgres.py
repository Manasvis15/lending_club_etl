import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from pathlib import Path
import joblib

load_dotenv()

BASE_DIR = Path("/home/hp/projects/lending_club_fin")
KMEANS_PATH = BASE_DIR / "models/kmeans_model.pkl"
CLUSTER_SCALER_PATH = BASE_DIR / "models/cluster_scaler.pkl"
CLUSTER_COLS_PATH = BASE_DIR / "models/cluster_cols.pkl"
XGB_PATH = BASE_DIR / "models/xgb_model.pkl"
FEATURES_PATH = BASE_DIR / "models/feature_columns.pkl"


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_to_postgres(df: pd.DataFrame, table: str = "lending_club_loans", chunksize: int = 5000) -> None:
    """Load transformed DataFrame into Postgres in chunks."""
    engine = get_engine()
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    print(f"[Load] {len(df):,} rows → {table}")


def load_predictions_to_postgres(df: pd.DataFrame) -> None:
    """
    Generate predictions and cluster labels from saved models
    and save them to the loan_predictions table.
    """
    from sklearn.preprocessing import LabelEncoder

    # --- Cluster assignment ---
    if KMEANS_PATH.exists():
        km_model = joblib.load(KMEANS_PATH)
        cluster_scaler = joblib.load(CLUSTER_SCALER_PATH)
        cluster_cols = joblib.load(CLUSTER_COLS_PATH)

        X_cluster = df[[c for c in cluster_cols if c in df.columns]].copy()
        X_cluster = X_cluster.apply(pd.to_numeric, errors="coerce").fillna(X_cluster.median())
        X_cluster_scaled = cluster_scaler.transform(X_cluster)
        cluster_labels = km_model.predict(X_cluster_scaled)
        print(f"[Predictions] Cluster labels assigned")
    else:
        cluster_labels = [-1] * len(df)
        print("[Predictions] No KMeans model found, cluster set to -1")

    # --- Default prediction ---
    if XGB_PATH.exists():
        xgb_model = joblib.load(XGB_PATH)
        feature_cols = joblib.load(FEATURES_PATH)

        # Prepare features the same way as training
        DROP_COLS = ["loan_status", "issue_d", "risk_band", "id"]
        X_pred = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()

        cat_cols = X_pred.select_dtypes(include="object").columns
        le = LabelEncoder()
        for col in cat_cols:
            X_pred[col] = le.fit_transform(X_pred[col].astype(str))

        X_pred = X_pred.apply(pd.to_numeric, errors="coerce").fillna(0)

        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_cols]

        y_pred = xgb_model.predict(X_pred)
        y_prob = xgb_model.predict_proba(X_pred)[:, 1]
        print(f"[Predictions] Default predictions generated")
    else:
        y_pred = [-1] * len(df)
        y_prob = [-1.0] * len(df)
        print("[Predictions] No XGBoost model found, predictions set to -1")

    # --- Actual default flag ---
    default_statuses = [
        "Charged Off", "Default",
        "Does not meet the credit policy. Status:Charged Off"
    ]
    y_actual = df["loan_status"].isin(default_statuses).astype(int) if "loan_status" in df.columns else [-1] * len(df)

    # --- Build predictions DataFrame ---
    pred_df = pd.DataFrame({
        "id": df["id"].astype(str) if "id" in df.columns else ["unknown"] * len(df),
        "is_default_actual": y_actual if isinstance(y_actual, list) else y_actual.values,
        "is_default_predicted": y_pred,
        "default_probability": y_prob,
        "risk_cluster": cluster_labels
    })

    engine = get_engine()
    pred_df.to_sql("loan_predictions", engine, if_exists="replace",
                   index=False, chunksize=5000, method="multi")
    print(f"[Predictions] {len(pred_df):,} rows → loan_predictions")


if __name__ == "__main__":
    df = pd.read_parquet(BASE_DIR / "data/processed/transformed.parquet")
    load_to_postgres(df)
    load_predictions_to_postgres(df)