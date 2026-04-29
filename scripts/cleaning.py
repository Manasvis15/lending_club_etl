import pandas as pd

KEEP_COLS = [
    "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "loan_status", "purpose", "dti", "delinq_2yrs", "fico_range_low",
    "fico_range_high", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "issue_d", "addr_state"
]

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns."""
    return df[[c for c in KEEP_COLS if c in df.columns]].copy()

def clean_pct_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip % from rate columns and convert to float."""
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("%", "").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clean_term(df: pd.DataFrame) -> pd.DataFrame:
    """Convert term '36 months' → 36."""
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    return df

def clean_emp_length(df: pd.DataFrame) -> pd.DataFrame:
    """Convert emp_length to numeric years."""
    mapping = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10
    }
    df["emp_length"] = df["emp_length"].map(mapping)
    return df

def drop_nulls(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop columns with more than threshold% missing values."""
    limit = len(df) * threshold
    return df.dropna(thresh=limit, axis=1)

def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run full cleaning pipeline."""
    df = select_columns(df)
    df = clean_pct_cols(df)
    df = clean_term(df)
    df = clean_emp_length(df)
    df = drop_nulls(df)
    df = df.dropna(subset=["loan_amnt", "loan_status", "annual_inc"])
    print(f"[Cleaning] Clean shape: {df.shape}")
    return df