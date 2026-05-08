**#🏦 The Anatomy of Default**
A Data Engineering Portfolio Project — Lending Club 2007–2018
An analytics project that ingests, cleans, transforms, and loads 1.8 million Lending Club loan records into PostgreSQL, orchestrated by Apache Airflow and visualized in Grafana. The central question: who fails to pay back a loan — and can we predict it from information available at origination?

**📐 Architecture**
Kaggle CSV
    │
    ▼
[ingestion.py]          Download → validate → save raw parquet
    │
    ▼
[cleaning.py]           Drop leakage/sparse cols, fix dtypes,
                        impute, deduplicate
    │
    ▼
[transformation.py]     Engineer features + build 4 mart aggregations
    │
    ▼
[load_to_postgres.py]   Bulk COPY to PostgreSQL (truncate → load)
    │
    ▼
Grafana Dashboard       "Anatomy of Default" story panels
Orchestrated end-to-end by Apache Airflow (etl_pipeline.py).

**🗂 Project Structure**
project/
├── airflow/
│   └── dags/
│       └── etl_pipeline.py       # Airflow DAG — task graph & scheduling
├── data/
│   ├── raw/                      # data_80.parquet (not in git)
│   └── processed/                # intermediate parquet - cleaned.parquet, transformed.parquet (not in git)
├── db/
│   └── init.sql                  # PostgreSQL schema (run once)
├── scripts/
│   ├── ingestion.py              # Download, validate, save raw data
│   ├── cleaning.py               # All cleaning steps as pure functions
│   ├── transformation.py         # Feature engineering 
│   └── load_to_postgres.py       # Bulk COPY parquet → Postgres tables
├── dashboards/
│   └── grafana_json/             # Dashboard
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis
├── .env                          # DB credentials (not in git)
├── requirements.txt
└── README.md

## 📦 Dataset

**Source:** [All Lending Club Loan Dataset from Kaggle - https://www.kaggle.com/datasets/wordsforthewise/lending-club/data]```

### Columns (151 original)

**Stack**

Python 3.11 — core language
Apache Airflow 2.9.1 — pipeline orchestration and scheduling
PostgreSQL 16 — data warehouse (port 5434)
Grafana 11 — dashboards and visualization
pandas / numpy — data processing
psycopg2 — PostgreSQL driver
WSL Ubuntu 24.04 — local Linux environment (no Docker)

**Pipeline**
raw CSV → ingestion → cleaning → transformation → load_to_postgres

Risk Band Definition
Low Risk FICO ≥ 740 and DTI < 15 
Medium Risk FICO ≥ 680 and DTI < 25
High Risk FICO ≥ 620 
Very High Risk - Everything else

**SETUP
Prerequisites**

WSL Ubuntu 24.04
Python 3.11
PostgreSQL 16 running on port 5434

**Install**
bashgit clone <your-repo-url>
cd lending_club_fin
python -m venv .venv
source .venv/bin/activate
python -m pip install "apache-airflow[postgres]==2.9.1" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.1/constraints-3.11.txt"
  
**Configure -** .env
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5434
DB_NAME=lending_club

**Create database**
bash - sudo -u postgres psql -p 5434

sql - 
CREATE DATABASE airflow_db;
CREATE DATABASE lending_club;
ALTER USER postgres WITH PASSWORD 'postgres';
Initialize Airflow

bash - 
export AIRFLOW_HOME=~/projects/lending_club_fin/airflow
airflow db migrate
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin \
  --email admin@example.com

Running the Pipeline
Every session
bash - 
cd ~/projects/lending_club_fin
source .venv/bin/activate
export AIRFLOW_HOME=~/projects/lending_club_fin/airflow
sudo service postgresql start

**Start Airflow**
bash - 
# Terminal 1
airflow webserver --port 8080

# Terminal 2
airflow scheduler

# Terminal 3 — trigger manually
airflow dags unpause lending_club_etl
airflow dags trigger lending_club_etl

Open http://localhost:8080 → login admin / admin

**Airflow DAG**
#Property: Value
DAG ID: lending_club_etl
Schedule: @daily
Tasks: ingestion → cleaning → transformation → load_to_postgres
Retries: 1 per task, 5 minute delay
Executor: LocalExecutor
Metadata DB: PostgreSQL (airflow_db)

**Grafana Dashboard**
#Start Grafana
bash - 
cd ~/grafana-v11.0.0
./bin/grafana-server &

Open http://localhost:3000 → login admin / admin

**Connect PostgreSQL**
#Setting: Value
Hostlocalhost:5434
Database: lending_club
User: grafana
Password: grafana
SSL Mode: disable

**Dashboard Structure**
Row 1 — Overview
Portfolio size, default rate, average interest rate, average borrower income, lending velocity over time, regional exposure heatmap.
Row 2 — Risk Analysis
Average rate vs default rate over time, default rate by risk band, default rate by grade, grade-wise loan volumes. Filterable by grade variable.
Row 3 — Borrower Profile
Default rate by loan purpose, default rate by employment length, average FICO by outcome, DTI vs FICO by risk band, default rate by state. Filterable by state variable.

**Key Findings**

Overall default rate across 1.8M loans is approximately 14%
Default rate increases monotonically from Grade A to Grade G
Very High Risk borrowers (low FICO, high DTI) default at significantly higher rates than Low Risk borrowers
Employment length shows a weak but present inverse relationship with default — shorter tenure correlates with slightly higher default rates
Loan purpose matters — small business and debt consolidation loans carry higher default rates than major purchase or home improvement loans
The platform's interest rate pricing closely tracks default rate over time, suggesting the risk was priced in but not eliminated


Data
Source: Lending Club Loan Data (Kaggle) — 1,808,532 rows after cleaning across 30 columns (24 original + 6 engineered).
Raw data excluded from version control due to file size.
