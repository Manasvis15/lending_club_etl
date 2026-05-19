import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/hp/projects/lending_club_fin")
import sys
sys.path.insert(0, str(BASE_DIR / "scripts"))
from cleaning import clean_pipeline
from transformation import transform_pipeline


def load_data():
    df_raw = pd.read_csv(BASE_DIR / "data/raw/data_80.csv", low_memory=False, dtype=str)
    df_clean = clean_pipeline(df_raw.copy())
    df = transform_pipeline(df_clean.copy())
    print(f"[Data] Shape: {df.shape}")
    return df


def prepare_features(df):
    TARGET = "is_default"
    DROP_COLS = ["loan_status", "issue_d", "risk_band"]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df.fillna(df.median())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"[Features] {X.shape[1]} features, {X.shape[0]} samples")
    print(f"[Features] Default rate: {y.mean()*100:.2f}%")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[Split] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)

    y_pred = lr.predict(X_test_sc)
    y_prob = lr.predict_proba(X_test_sc)[:, 1]

    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return lr, scaler, y_pred, y_prob


def train_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return rf, y_pred, y_prob


def train_xgboost(X_train, X_test, y_train, y_test):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="auc",
        verbosity=0
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    print("\n=== XGBoost ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

    return xgb, y_pred, y_prob


def plot_roc_curves(y_test, probs_dict):
    plt.figure(figsize=(10, 6))
    for name, prob in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/roc_curves.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/confusion_matrix.png")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=importance.values, y=importance.index, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "dashboards/grafana_screenshots/feature_importance.png")
    plt.show()


def model_comparison(y_test, results):
    summary = []
    for name, (y_pred, y_prob) in results.items():
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        summary.append({
            "Model": name,
            "ROC-AUC": round(auc, 4),
            "Precision (Default)": round(report["1"]["precision"], 4),
            "Recall (Default)": round(report["1"]["recall"], 4),
            "F1 (Default)": round(report["1"]["f1-score"], 4),
        })
    summary_df = pd.DataFrame(summary).set_index("Model")
    print("\n=== Model Comparison ===")
    print(summary_df.to_string())
    return summary_df


if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    lr_model, scaler, lr_pred, lr_prob = train_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_pred, rf_prob = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_pred, xgb_prob = train_xgboost(X_train, X_test, y_train, y_test)

    plot_roc_curves(y_test, {
        "Logistic Regression": lr_prob,
        "Random Forest": rf_prob,
        "XGBoost": xgb_prob
    })
    plot_confusion_matrix(y_test, xgb_pred, "XGBoost")
    plot_feature_importance(xgb_model, X.columns)
    model_comparison(y_test, {
        "Logistic Regression": (lr_pred, lr_prob),
        "Random Forest": (rf_pred, rf_prob),
        "XGBoost": (xgb_pred, xgb_prob)
    })