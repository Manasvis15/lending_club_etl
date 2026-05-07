# 📊 Lending Club Dataset
---

## 🏗️ Architecture

```
Raw Data (Kaggle)
       ↓
   Ingestion
   (Python)
       ↓
Cleaning & Transformation
   (Python Scripts)
       ↓
Apache Airflow
(ETL Orchestration)
       ↓
 Processed Data
 (Parquet/CSV)
       ↓
   PostgreSQL
  (finance_db)
       ↓
  Grafana
(Dashboards)
```

---

## 📁 Project Structure

```
finance-economic-pipeline/
│
├── airflow/
│   └── dags/
│       └── etl_pipeline.py          # Airflow DAG — orchestrates full ETL
│
├── data/
│   ├── raw/
│   │   ├── finance_economic_dataset.csv   # Original Kaggle dataset
│   │   ├── data_80.csv                    # 80% training split
│   │   └── data_20.csv                    # 20% holdout split
│   └── processed/
│       └── processed.parquet              # Cleaned & transformed data
│
├── db/
│   └── init.sql                     # PostgreSQL table schema
│
├── notebooks/
│   ├── eda.ipynb                    # Exploratory data analysi
│
├── scripts/
│   └── load_to_postgres.py          # Loads processed data into PostgreSQL
│   ├── ingestion.py                 # load_csv, load_parquet functions
│   ├── cleaning.py                  # Cleaning functions
│   ├── transformation.py            # Transformation functions
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**Source:** [All Lending Club Loan Dataset from Kaggle]```

### Columns (151 original)

### Engineered Features (10 additional)
| Feature | Description |
|---|---|
| `Daily_Return (%)` | % gain/loss between open and close price |
| `Misery_Index` | Inflation Rate + Unemployment Rate |
| `Real_Interest_Rate (%)` | Interest Rate − Inflation Rate |
| `Volatility_Band (%)` | Daily high-low range as % of close price |
| `Economic_Health_Score` | Composite of GDP, confidence and profits |
| `Year` | Extracted from Date |
| `Month` | Extracted from Date |
| `Quarter` | Extracted from Date |
| `Month_Name` | Extracted from Date |
| `Stock_Index_encoded` | Label encoded Stock Index |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Scikit-learn | ML models and preprocessing |
| Matplotlib / Seaborn | Visualisation |
| Statsmodels / SciPy | Statistical analysis |
| PostgreSQL 18 | Data warehouse |
| SQLAlchemy + psycopg2 | PostgreSQL connection |
| Apache Airflow 3.x | Pipeline orchestration |
| Grafana | Dashboard and visualisation |
| Git / GitHub | Version control |

---

## 🚀 How to Run

### Prerequisites
- Python 3.12
- PostgreSQL 18
- Apache Airflow (via WSL/Debian)
- Git

### 1. Clone the repository
```bash
git clone https://github.com/Manasvis15/finance-economic-pipeline.git
cd finance-economic-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle]
```
data/raw/data_80.csv
```

### 4. Set up PostgreSQL
```bash
# Create database
psql -U postgres
CREATE DATABASE lending_club_loans;
\q

# Create table
psql -U postgres -d lending_club_loans -f db/init.sql
```

### 5. Run the notebook
```
1. notebooks/eda.ipynb
```

### 6. Load data to PostgreSQL
```bash
python scripts/load_to_postgres.py
```

### 7. Run Airflow (via WSL/Debian)
```bash
# Activate virtual environment
source ~/airflow-venv/bin/activate

# Start Airflow
export AIRFLOW_HOME=~/airflow
airflow standalone
```
Open `http://localhost:8081` and trigger `finance_etl_pipeline` DAG.

---

## 🔄 ETL Pipeline

The Airflow DAG `lending_club_pipeline` automates the full pipeline:

```
Task 1: Ingestion
        ↓
Task 2: Cleaning & Transformation
        ↓
Task 3: Load to PostgreSQL
```

**Schedule:** Daily  
**DAG file:** `airflow/dags/etl_pipeline.py`

---

## 🧹 Data Cleaning Steps

| Step | Method |
|---|---|
| Null values | Median fill for numeric, mode fill for categorical |
| Duplicates | Dropped and index reset |
| Date format | Standardised to YYYY-MM-DD |
| Text formatting | Stripped and uppercased |
| Outliers | IQR capping (1.5 × IQR) |

---

## 📊 EDA Highlights

- **Missing value analysis** — bar charts of null percentages per column
- **Distribution plots** — histograms with KDE for all 34 columns
- **Correlation heatmap** — relationships between all numeric indicators
- **Outlier detection** — IQR-based boxplots per column
- **Skewness & kurtosis** — normality assessment
- **Time series trends** — all indicators plotted over 2000–present
- **EDA summary report** — consolidated findings

---

## 🗄️ PostgreSQL Schema

**Database:** `lending_club_loans`  
**Table:** `lending_club`  
**Rows:** 1808561  
**Columns:** 151  

```sql
-- Connect
psql -U postgres -d lending_club_loans

-- Check row count
SELECT COUNT(*) FROM lending_club_loans;

-- Check date range
SELECT MIN("Date"), MAX("Date") FROM lending_club_loans;
```

---

## ⚙️ Configuration

Update credentials in `scripts/load_to_postgres.py`:
```python
DB_CONFIG = {
    'host'    : 'localhost',
    'port'    : 5434,
    'database': 'lending_club_loans',
    'user'    : 'postgres',
    'password': 'postgres'
}
```

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
sqlalchemy
psycopg2-binary
pyarrow
apache-airflow
jupyter
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Manasvi**  
GitHub: [@Manasvis15](https://github.com/Manasvis15)

---

## 📄 License

This project is for educational and portfolio purposes.