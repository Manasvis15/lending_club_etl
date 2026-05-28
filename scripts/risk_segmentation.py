import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sqlalchemy import create_engine
from dotenv import load_dotenv
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
BASE_DIR = Path("/home/hp/projects/lending_club_fin")
KMEANS_PATH = BASE_DIR / "models/kmeans_model.pkl"
CLUSTER_SCALER_PATH = BASE_DIR / "models/cluster_scaler.pkl"
CLUSTER_COLS_PATH = BASE_DIR / "models/cluster_cols.pkl"

import sys
sys.path.insert(0, str(BASE_DIR / "scripts"))
from cleaning import clean_pipeline
from transformation import transform_pipeline


def get_engine():
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)


def load_data():
    df_raw = pd.read_csv(BASE_DIR / "data/raw/data_80.csv", low_memory=False, dtype=str)
    df_clean = clean_pipeline(df_raw.copy())
    df = transform_pipeline(df_clean.copy())
    print(f"[Data] Shape: {df.shape}")
    return df


def prepare_clustering_features(df):
    CLUSTER_FEATURES = [
        "loan_amnt", "int_rate", "annual_inc", "dti",
        "fico_mid", "open_acc", "revol_util", "income_to_loan_ratio"
    ]
    cluster_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X = df[cluster_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[Clustering] Features: {cluster_cols}")
    return X, X_scaled, cluster_cols, scaler


def plot_elbow(X_scaled, max_k=10):
    inertias = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, marker="o", color="steelblue")
    plt.title("Elbow Method — Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/elbow_curve.png")
    plt.show()


def plot_silhouette(X_scaled, max_k=10):
    scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(10, 5))
    plt.plot(k_range, scores, marker="o", color="green")
    plt.title("Silhouette Score by K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/silhouette_scores.png")
    plt.show()

    best_k = list(k_range)[np.argmax(scores)]
    print(f"[Silhouette] Best K: {best_k}")
    return best_k


def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    print(f"[KMeans] Cluster distribution:\n{pd.Series(labels).value_counts().sort_index()}")
    return km, labels


def save_clustering_model(km_model, scaler, cluster_cols):
    """Save KMeans model, scaler and feature columns to disk."""
    KMEANS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(km_model, KMEANS_PATH)
    joblib.dump(scaler, CLUSTER_SCALER_PATH)
    joblib.dump(cluster_cols, CLUSTER_COLS_PATH)
    print(f"[Model] KMeans saved to {KMEANS_PATH.parent}")


def predict_cluster_single(input_dict: dict) -> dict:
    """
    Live cluster assignment for a single borrower.

    Usage:
        result = predict_cluster_single({
            "loan_amnt": 15000,
            "int_rate": 12.5,
            "annual_inc": 60000,
            "dti": 18.5,
            "fico_mid": 682,
            "open_acc": 8,
            "revol_util": 45.0,
            "income_to_loan_ratio": 0.25
        })
        print(result)
    """
    if not KMEANS_PATH.exists():
        raise FileNotFoundError("KMeans model not found. Run training first.")

    km_model = joblib.load(KMEANS_PATH)
    scaler = joblib.load(CLUSTER_SCALER_PATH)
    cluster_cols = joblib.load(CLUSTER_COLS_PATH)

    input_df = pd.DataFrame([input_dict])
    for col in cluster_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[cluster_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    X_scaled = scaler.transform(input_df)
    cluster = int(km_model.predict(X_scaled)[0])

    # Interpret cluster
    fico = input_dict.get("fico_mid", 0)
    dti = input_dict.get("dti", 0)

    if fico >= 720 and dti < 15:
        risk_label = "Prime Borrowers — Low Risk"
    elif fico >= 680 and dti < 25:
        risk_label = "Near-Prime — Medium Risk"
    elif fico >= 640:
        risk_label = "Subprime — High Risk"
    else:
        risk_label = "Deep Subprime — Very High Risk"

    result = {
        "risk_cluster": cluster,
        "risk_label": risk_label
    }

    print("\n=== Live Cluster Assignment ===")
    print(f"  Cluster       : {cluster}")
    print(f"  Risk Label    : {risk_label}")

    return result


def plot_pca_clusters(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.5, s=5)
    plt.colorbar(scatter, label="Cluster")
    plt.title("KMeans Clusters (PCA 2D Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/pca_clusters.png")
    plt.show()
    print(f"[PCA] Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")


def plot_cluster_profiles(X_cluster, cluster_cols, labels):
    X_cluster = X_cluster.copy()
    X_cluster["cluster"] = labels

    profile = X_cluster.groupby("cluster")[cluster_cols].mean()
    print("\n=== Cluster Profiles ===")
    print(profile.round(2).to_string())

    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())

    plt.figure(figsize=(14, 6))
    sns.heatmap(profile_norm.T, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5)
    plt.title("Normalized Cluster Feature Profiles")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/cluster_profiles.png")
    plt.show()

    return profile


def plot_default_by_cluster(df, cluster_labels):
    default_statuses = [
        "Charged Off", "Default",
        "Does not meet the credit policy. Status:Charged Off"
    ]
    df = df.copy()
    df["cluster"] = cluster_labels
    df["is_default"] = df["loan_status"].isin(default_statuses).astype(int)

    default_by_cluster = df.groupby("cluster")["is_default"].mean() * 100

    plt.figure(figsize=(10, 5))
    sns.barplot(x=default_by_cluster.index, y=default_by_cluster.values, palette="Reds")
    plt.title("Default Rate by Cluster (%)")
    plt.xlabel("Cluster")
    plt.ylabel("Default Rate (%)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/default_by_cluster.png")
    plt.show()


def interpret_clusters(cluster_profiles):
    print("\n=== CLUSTER INTERPRETATION ===\n")
    for cluster_id, row in cluster_profiles.iterrows():
        fico = row.get("fico_mid", None)
        dti = row.get("dti", None)
        inc = row.get("annual_inc", None)
        loan = row.get("loan_amnt", None)

        if fico is not None and dti is not None:
            if fico >= 720 and dti < 15:
                label = "Prime Borrowers — Low Risk"
            elif fico >= 680 and dti < 25:
                label = "Near-Prime — Medium Risk"
            elif fico >= 640:
                label = "Subprime — High Risk"
            else:
                label = "Deep Subprime — Very High Risk"
        else:
            label = "Unclassified"

        print(f"Cluster {cluster_id}: {label}")
        if fico is not None:
            print(f"  Avg FICO:   {fico:.0f}")
        if dti is not None:
            print(f"  Avg DTI:    {dti:.1f}%")
        if inc is not None:
            print(f"  Avg Income: ${inc:,.0f}")
        if loan is not None:
            print(f"  Avg Loan:   ${loan:,.0f}\n")


if __name__ == "__main__":
    df = load_data()
    X_cluster, X_scaled, cluster_cols, scaler = prepare_clustering_features(df.copy())

    plot_elbow(X_scaled)
    best_k = plot_silhouette(X_scaled)

    km_model, cluster_labels = fit_kmeans(X_scaled, best_k)

    # Save clustering model for live prediction
    save_clustering_model(km_model, scaler, cluster_cols)

    plot_pca_clusters(X_scaled, cluster_labels)
    cluster_profiles = plot_cluster_profiles(X_cluster, cluster_cols, cluster_labels)
    plot_default_by_cluster(df, cluster_labels)
    interpret_clusters(cluster_profiles)

    # Example live cluster prediction
    predict_cluster_single({
        "loan_amnt": 15000,
        "int_rate": 12.5,
        "annual_inc": 60000,
        "dti": 18.5,
        "fico_mid": 682,
        "open_acc": 8,
        "revol_util": 45.0,
        "income_to_loan_ratio": 0.25
    })