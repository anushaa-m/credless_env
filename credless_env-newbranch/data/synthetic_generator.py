"""
Synthetic data generator for the FinVerse compatibility pipeline.

Design guardrails:
  - Generates only applicant features by default.
  - Does not attach oracle-derived labels unless `include_target=True`.
  - Synthetic data is suitable for smoke tests and fallback flows, not for
    benchmark claims against the same oracle that labeled it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_samples: int = 12000,
    seed: int = 42,
    include_target: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_samples)
    marital_status = rng.choice(
        ["single", "married", "divorced", "widowed"],
        n_samples,
        p=[0.30, 0.52, 0.13, 0.05],
    )
    dependents_count = rng.choice(
        [0, 1, 2, 3, 4, 5],
        n_samples,
        p=[0.28, 0.22, 0.28, 0.14, 0.05, 0.03],
    )
    number_of_dependents = dependents_count.copy()

    monthly_income = np.clip(
        rng.lognormal(mean=8.7, sigma=0.7, size=n_samples),
        1000,
        200000,
    )
    debt_ratio = np.clip(rng.beta(a=2, b=5, size=n_samples) * 1.2, 0, 2.0)
    revolving_util = np.clip(rng.beta(a=2, b=3, size=n_samples), 0, 1)

    late_30_59 = rng.poisson(lam=0.4, size=n_samples)
    late_60_89 = rng.poisson(lam=0.15, size=n_samples)
    late_90_plus = rng.poisson(lam=0.08, size=n_samples)

    open_credit_lines = rng.integers(0, 25, n_samples)
    real_estate_loans = rng.integers(0, 5, n_samples)

    years_at_address = rng.integers(0, 30, n_samples)
    months_employed = rng.integers(0, 360, n_samples)
    employment_type = rng.choice(
        ["salaried", "self_employed", "contract", "unemployed", "retired"],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.10, 0.05],
    )

    bank_account_age_months = rng.integers(1, 360, n_samples)
    avg_monthly_balance = np.clip(
        rng.lognormal(mean=8.2, sigma=0.9, size=n_samples),
        0,
        500000,
    )
    negative_balance_days = rng.integers(0, 30, n_samples)
    overdraft_count = rng.poisson(lam=0.8, size=n_samples)

    monthly_upi_spend = np.clip(
        rng.lognormal(mean=7.5, sigma=0.8, size=n_samples),
        0,
        100000,
    )
    active_txn_days = rng.integers(1, 30, n_samples)
    failed_txn_ratio = np.clip(rng.beta(a=1.5, b=8, size=n_samples), 0, 1)
    late_night_txn_ratio = np.clip(rng.beta(a=1, b=9, size=n_samples), 0, 1)

    utility_payment_ratio = np.clip(rng.beta(a=7, b=2, size=n_samples), 0, 1)
    rent_payment_regular = rng.choice([0, 1], n_samples, p=[0.25, 0.75])
    emi_payment_ratio = np.clip(rng.beta(a=6, b=2, size=n_samples), 0, 1)

    vehicle_owned = rng.choice([0, 1], n_samples, p=[0.45, 0.55])
    gold_value_estimate = np.where(
        rng.random(n_samples) > 0.4,
        rng.lognormal(mean=9.5, sigma=0.8, size=n_samples),
        0,
    )
    fd_amount = np.where(
        rng.random(n_samples) > 0.55,
        rng.lognormal(mean=10.5, sigma=1.0, size=n_samples),
        0,
    )
    emergency_savings_months = rng.integers(0, 24, n_samples)
    property_owned = rng.choice([0, 1], n_samples, p=[0.60, 0.40])

    income_variability_score = np.clip(rng.beta(a=2, b=4, size=n_samples), 0, 1)
    location_risk_index = np.clip(rng.beta(a=2, b=3, size=n_samples), 0, 1)

    current_medical_condition = rng.choice([0, 1], n_samples, p=[0.80, 0.20])
    medical_debt = np.where(
        current_medical_condition == 1,
        rng.lognormal(mean=9, sigma=1, size=n_samples),
        0,
    )
    gov_scheme_enrollment = rng.choice([0, 1], n_samples, p=[0.65, 0.35])

    business_age = np.where(
        employment_type == "self_employed",
        rng.integers(0, 240, n_samples),
        0,
    )
    business_type_risk = np.clip(rng.beta(a=2, b=4, size=n_samples), 0, 1)
    monthly_revenue = np.where(
        employment_type == "self_employed",
        np.clip(rng.lognormal(mean=9.5, sigma=0.9, size=n_samples), 0, 500000),
        0,
    )
    profit_margin = np.where(
        monthly_revenue > 0,
        np.clip(rng.beta(a=3, b=5, size=n_samples), 0, 1),
        0,
    )

    df = pd.DataFrame(
        {
            "age": age,
            "marital_status": marital_status,
            "dependents_count": dependents_count,
            "numberofdependents": number_of_dependents,
            "monthlyincome": monthly_income.round(2),
            "debtratio": debt_ratio.round(4),
            "revolvingutilizationofunsecuredlines": revolving_util.round(4),
            "numberoftime30-59dayspastduenotworse": late_30_59,
            "numberoftime60-89dayspastduenotworse": late_60_89,
            "numberoftimes90dayslate": late_90_plus,
            "numberofopencreditlinesandloans": open_credit_lines,
            "numberrealestateloansorlines": real_estate_loans,
            "years_at_address": years_at_address,
            "months_employed": months_employed,
            "employment_type": employment_type,
            "bank_account_age_months": bank_account_age_months,
            "avg_monthly_balance": avg_monthly_balance.round(2),
            "negative_balance_days": negative_balance_days,
            "overdraft_count": overdraft_count,
            "monthly_upi_spend": monthly_upi_spend.round(2),
            "active_txn_days": active_txn_days,
            "failed_txn_ratio": failed_txn_ratio.round(4),
            "late_night_txn_ratio": late_night_txn_ratio.round(4),
            "utility_payment_ratio": utility_payment_ratio.round(4),
            "rent_payment_regular": rent_payment_regular,
            "emi_payment_ratio": emi_payment_ratio.round(4),
            "vehicle_owned": vehicle_owned,
            "gold_value_estimate": gold_value_estimate.round(2),
            "fd_amount": fd_amount.round(2),
            "emergency_savings_months": emergency_savings_months,
            "property_owned": property_owned,
            "income_variability_score": income_variability_score.round(4),
            "location_risk_index": location_risk_index.round(4),
            "current_medical_condition": current_medical_condition,
            "medical_debt": medical_debt.round(2),
            "gov_scheme_enrollment": gov_scheme_enrollment,
            "business_age": business_age,
            "business_type_risk": business_type_risk.round(4),
            "monthly_revenue": monthly_revenue.round(2),
            "profit_margin": profit_margin.round(4),
        }
    )

    if include_target:
        from server.oracle import oracle_decision

        targets = []
        for _, row in df.iterrows():
            result = oracle_decision(row.to_dict())
            targets.append(1 if result["decision"] == "approve" else 0)
        df["target"] = targets
        print(
            f"[synthetic_generator] Generated {len(df)} rows with target | "
            f"Approve rate: {df['target'].mean():.2%}"
        )
    else:
        print(f"[synthetic_generator] Generated {len(df)} unlabeled synthetic rows")

    return df
