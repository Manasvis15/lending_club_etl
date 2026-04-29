
DROP TABLE IF EXISTS lending_club_loans;

CREATE TABLE lending_club_loans (
    -- Loan Identifiers
    id                          VARCHAR(20) PRIMARY KEY,
    member_id                   VARCHAR(20),

    -- Loan Info
    loan_amnt                   NUMERIC(12,2),
    funded_amnt                 NUMERIC(12,2),
    funded_amnt_inv             NUMERIC(12,2),
    term                        VARCHAR(20),
    int_rate                    NUMERIC(6,3),
    installment                 NUMERIC(10,2),
    grade                       VARCHAR(5),
    sub_grade                   VARCHAR(5),
    purpose                     VARCHAR(50),
    title                       VARCHAR(100),
    initial_list_status         VARCHAR(5),
    disbursement_method         VARCHAR(20),

    -- Borrower Info
    emp_title                   VARCHAR(100),
    emp_length                  VARCHAR(20),
    home_ownership              VARCHAR(20),
    annual_inc                  NUMERIC(15,2),
    verification_status         VARCHAR(30),
    addr_state                  VARCHAR(5),
    zip_code                    VARCHAR(10),
    dti                         NUMERIC(8,3),

    -- Dates
    issue_d                     VARCHAR(20),
    earliest_cr_line            VARCHAR(20),
    last_pymnt_d                VARCHAR(20),
    next_pymnt_d                VARCHAR(20),
    last_credit_pull_d          VARCHAR(20),

    -- Loan Status
    loan_status                 VARCHAR(50),
    pymnt_plan                  VARCHAR(5),

    -- Credit Info
    fico_range_low              NUMERIC(6,2),
    fico_range_high             NUMERIC(6,2),
    delinq_2yrs                 NUMERIC(6,2),
    inq_last_6mths              NUMERIC(6,2),
    mths_since_last_delinq      NUMERIC(6,2),
    mths_since_last_record      NUMERIC(6,2),
    open_acc                    NUMERIC(6,2),
    pub_rec                     NUMERIC(6,2),
    revol_bal                   NUMERIC(15,2),
    revol_util                  NUMERIC(8,3),
    total_acc                   NUMERIC(6,2),

    -- Payment Info
    out_prncp                   NUMERIC(12,2),
    out_prncp_inv               NUMERIC(12,2),
    total_pymnt                 NUMERIC(12,2),
    total_pymnt_inv             NUMERIC(12,2),
    total_rec_prncp             NUMERIC(12,2),
    total_rec_int               NUMERIC(12,2),
    total_rec_late_fee          NUMERIC(10,2),
    recoveries                  NUMERIC(12,2),
    collection_recovery_fee     NUMERIC(10,2),
    last_pymnt_amnt             NUMERIC(12,2),

    -- Hardship
    hardship_flag               VARCHAR(5),
    hardship_type               VARCHAR(100),
    hardship_reason             VARCHAR(100),
    hardship_status             VARCHAR(50),
    deferral_term               NUMERIC(6,2),
    hardship_amount             NUMERIC(12,2),
    hardship_start_date         VARCHAR(20),
    hardship_end_date           VARCHAR(20),
    payment_plan_start_date     VARCHAR(20),
    hardship_length             NUMERIC(6,2),
    hardship_dpd                NUMERIC(6,2),
    hardship_loan_status        VARCHAR(50),
    orig_projected_additional_accrued_interest NUMERIC(12,2),
    hardship_payoff_balance_amount NUMERIC(12,2),
    hardship_last_payment_amount   NUMERIC(12,2),

    -- Debt Settlement
    debt_settlement_flag        VARCHAR(5),
    debt_settlement_flag_date   VARCHAR(20),
    settlement_status           VARCHAR(50),
    settlement_date             VARCHAR(20),
    settlement_amount           NUMERIC(12,2),
    settlement_percentage       NUMERIC(8,3),
    settlement_term             NUMERIC(6,2)
);