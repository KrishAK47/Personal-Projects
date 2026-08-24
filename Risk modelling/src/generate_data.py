"""
generate_data.py
-----------------
Generates a synthetic but realistic consumer-loan portfolio dataset for the
"Risk Modelling: Predicting Loan Defaulters" case study.

The dataset mimics fields typically found in retail-banking loan books
(credit bureau attributes + loan-application attributes) and simulates a
binary default label whose probability depends on a weighted, non-linear
combination of the features plus random noise -- so the resulting dataset
has realistic, learnable-but-imperfect structure (similar to real credit
risk data, e.g. the LendingClub / German Credit datasets).

No real customer data is used anywhere in this project.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 15000

rng = np.random.default_rng(RANDOM_SEED)


def generate_loan_book(n=N_SAMPLES) -> pd.DataFrame:
    # ---- Demographic / employment features -----------------------------
    age = np.clip(rng.normal(40, 11, n), 20, 75).round(0)
    employment_length = np.clip(rng.exponential(5, n), 0, 40).round(1)

    home_ownership = rng.choice(
        ["MORTGAGE", "RENT", "OWN"], size=n, p=[0.45, 0.40, 0.15]
    )

    annual_income = np.clip(
        rng.lognormal(mean=10.9, sigma=0.45, size=n), 18000, 350000
    ).round(0)

    # ---- Credit bureau features ------------------------------------------
    credit_score = np.clip(rng.normal(680, 65, n), 300, 850).round(0)
    credit_utilization = np.clip(rng.beta(2, 3, n), 0, 1).round(3)
    delinquencies_2yrs = rng.poisson(0.35, n)
    months_since_last_delinq = np.where(
        delinquencies_2yrs > 0, rng.integers(1, 48, n), 999
    )
    open_credit_lines = np.clip(rng.poisson(7, n), 1, 30)
    existing_loans_count = np.clip(rng.poisson(1.5, n), 0, 10)

    # ---- Loan application features ---------------------------------------
    purpose = rng.choice(
        [
            "debt_consolidation",
            "credit_card",
            "home_improvement",
            "business",
            "medical",
            "auto",
            "education",
            "other",
        ],
        size=n,
        p=[0.30, 0.18, 0.12, 0.10, 0.08, 0.10, 0.07, 0.05],
    )

    loan_amount = np.clip(rng.lognormal(mean=9.2, sigma=0.55, size=n), 1000, 60000).round(0)
    loan_term_months = rng.choice([36, 60], size=n, p=[0.65, 0.35])

    debt_to_income = np.clip(
        (loan_amount / 12 * (loan_term_months / 36)) / (annual_income / 12 + 1)
        + rng.normal(0.15, 0.08, n),
        0.0,
        0.75,
    ).round(3)

    # Interest rate roughly priced off credit score + DTI (as a real lender would)
    base_rate = 22 - (credit_score - 300) / 550 * 16
    interest_rate = np.clip(
        base_rate + debt_to_income * 8 + rng.normal(0, 1.5, n), 5.0, 29.99
    ).round(2)

    # ---- Simulate ground-truth default probability -----------------------
    # Standardize a few key drivers for a clean logistic combination
    z_credit = (credit_score - 680) / 65
    z_dti = (debt_to_income - 0.25) / 0.15
    z_util = (credit_utilization - 0.4) / 0.2
    z_income = (np.log(annual_income) - 10.9) / 0.45
    z_emp = (employment_length - 5) / 5

    logit = (
        -3.45
        - 1.55 * z_credit
        + 0.95 * z_dti
        + 0.55 * z_util
        + 0.55 * delinquencies_2yrs
        - 0.25 * z_income
        - 0.20 * z_emp
        + 0.30 * (loan_term_months == 60).astype(int)
        + 0.18 * (interest_rate - 13) / 5
        + 0.15 * (existing_loans_count > 3).astype(int)
        + rng.normal(0, 0.65, n)  # unexplained noise, keeps the problem realistic
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{100000 + i}" for i in range(n)],
            "age": age.astype(int),
            "annual_income": annual_income,
            "employment_length_years": employment_length,
            "home_ownership": home_ownership,
            "credit_score": credit_score.astype(int),
            "credit_utilization": credit_utilization,
            "delinquencies_2yrs": delinquencies_2yrs,
            "months_since_last_delinquency": months_since_last_delinq,
            "open_credit_lines": open_credit_lines,
            "existing_loans_count": existing_loans_count,
            "loan_purpose": purpose,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate,
            "debt_to_income_ratio": debt_to_income,
            "default": default,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_loan_book()
    out_path = "data/loan_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(df.head())
