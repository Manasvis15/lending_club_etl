import pandas as pd
import numpy as np

DEFAULT_STATUSES = [
    "Charged Off", "Default",
    "Does not meet the credit policy. Status:Charged Off"
]

def create_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary target: 1 = defaulted."""
    df["is_default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return df

def create_risk_band(df: pd.DataFrame) -> pd.DataFrame:
    """Segment loans into risk tiers using FICO + DTI."""
    df["fico_range_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
    df["dti"] = pd.to_numeric(df["dti"], errors="coerce")
    conditions = [
        (df["fico_range_low"] >= 740) & (df["dti"] < 15),
        (df["fico_range_low"] >= 680) & (df["dti"] < 25),
        (df["fico_range_low"] >= 620),
    ]
    choices = ["Low Risk", "Medium Risk", "High Risk"]
    df["risk_band"] = np.select(conditions, choices, default="Very High Risk")
    return df

def create_income_to_loan_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_amnt"] = pd.to_numeric(df["loan_amnt"], errors="coerce")
    df["annual_inc"] = pd.to_numeric(df["annual_inc"], errors="coerce")
    df["income_to_loan_ratio"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    return df

def create_fico_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    df["fico_range_low"] = pd.to_numeric(df["fico_range_low"], errors="coerce")
    df["fico_range_high"] = pd.to_numeric(df["fico_range_high"], errors="coerce")
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df

def create_issue_year_month(df: pd.DataFrame) -> pd.DataFrame:
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_d"].dt.year
    df["issue_month"] = df["issue_d"].dt.month
    return df

def transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all transformations."""
    df = create_default_flag(df)
    df = create_risk_band(df)
    df = create_income_to_loan_ratio(df)
    df = create_fico_midpoint(df)
    df = create_issue_year_month(df)
    df.to_parquet("/home/hp/projects/lending_club_fin/data/processed/transformed.parquet", index=False)
    print(f"[Transform] Final shape: {df.shape}")
    return df