import sys
sys.path.insert(0, "/home/hp/projects/lending_club_fin/scripts")

from default_prediction import predict_single
from risk_segmentation import predict_cluster_single

print("=== Lending Club Live Predictor ===\n")

loan_amnt     = float(input("Loan Amount ($): "))
int_rate      = float(input("Interest Rate (%): "))
annual_inc    = float(input("Annual Income ($): "))
dti           = float(input("DTI: "))
fico_low      = float(input("FICO Range Low: "))
fico_high     = float(input("FICO Range High: "))
grade         = input("Grade (A-G): ")
emp_length    = float(input("Employment Length (years): "))
term          = float(input("Term (36 or 60): "))
revol_util    = float(input("Revolving Utilization (%): "))
open_acc      = float(input("Open Accounts: "))
pub_rec       = float(input("Public Records: "))
delinq_2yrs   = float(input("Delinquencies (2yrs): "))
revol_bal     = float(input("Revolving Balance ($): "))
total_acc     = float(input("Total Accounts: "))

borrower = {
    "loan_amnt": loan_amnt,
    "int_rate": int_rate,
    "annual_inc": annual_inc,
    "dti": dti,
    "fico_range_low": fico_low,
    "fico_range_high": fico_high,
    "grade": grade,
    "emp_length": emp_length,
    "term": term,
    "revol_util": revol_util,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "delinq_2yrs": delinq_2yrs,
    "revol_bal": revol_bal,
    "total_acc": total_acc
}

predict_single(borrower)

predict_cluster_single({
    "loan_amnt": loan_amnt,
    "int_rate": int_rate,
    "annual_inc": annual_inc,
    "dti": dti,
    "fico_mid": (fico_low + fico_high) / 2,
    "open_acc": open_acc,
    "revol_util": revol_util,
    "income_to_loan_ratio": loan_amnt / annual_inc if annual_inc > 0 else 0
})