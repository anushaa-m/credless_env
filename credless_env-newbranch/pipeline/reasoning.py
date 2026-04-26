from __future__ import annotations

from typing import Dict


def generate_reasoning(applicant: Dict[str, object], oracle_result: Dict[str, object]) -> str:
    """
    Produces a deterministic, feature-grounded reasoning string for a lending
    decision. This mirrors the raw-row oracle in readable English and supports
    the project's mixed label formats.
    """
    positives: list[str] = []
    negatives: list[str] = []

    late_90 = int(float(applicant.get("numberoftimes90dayslate", 0) or 0))
    late_60 = int(float(applicant.get("numberoftime60-89dayspastduenotworse", 0) or 0))
    late_30 = int(float(applicant.get("numberoftime30-59dayspastduenotworse", 0) or 0))

    if late_90 >= 3:
        negatives.append(f"severely delinquent payment history ({late_90} times 90+ days late)")
    elif late_90 >= 1:
        negatives.append(f"prior 90-day late payments ({late_90} occurrence(s))")

    if late_60 >= 2:
        negatives.append(f"multiple 60-89 day late payments ({late_60})")
    if late_30 >= 3:
        negatives.append(f"recurring 30-59 day delays ({late_30})")
    if late_90 == 0 and late_60 == 0 and late_30 == 0:
        positives.append("clean credit repayment history with no delinquencies")

    debt_ratio = float(applicant.get("debtratio", 0) or 0)
    if debt_ratio > 1.0:
        negatives.append(f"dangerously high debt-to-income ratio ({debt_ratio:.2f})")
    elif debt_ratio > 0.60:
        negatives.append(f"elevated debt ratio ({debt_ratio:.2f})")
    elif debt_ratio < 0.25:
        positives.append(f"healthy debt-to-income ratio ({debt_ratio:.2f})")

    rev_util = float(applicant.get("revolvingutilizationofunsecuredlines", 0) or 0)
    if rev_util > 0.80:
        negatives.append(f"near-maxed revolving credit utilization ({rev_util:.0%})")
    elif rev_util > 0.60:
        negatives.append(f"high revolving credit utilization ({rev_util:.0%})")
    elif rev_util < 0.20:
        positives.append(f"low revolving credit utilization ({rev_util:.0%})")

    income = float(applicant.get("monthlyincome", 0) or 0)
    if income >= 50_000:
        positives.append(f"strong monthly income (INR {income:,.0f})")
    elif income >= 10_000:
        positives.append(f"adequate monthly income (INR {income:,.0f})")
    elif income < 3_000:
        negatives.append(f"very low monthly income (INR {income:,.0f})")

    variability = float(applicant.get("income_variability_score", 0) or 0)
    if variability > 0.60:
        negatives.append("high income variability indicating unstable earnings")

    months_emp = int(float(applicant.get("months_employed", 0) or 0))
    if months_emp >= 24:
        positives.append(f"stable employment tenure ({months_emp} months)")
    elif months_emp < 6:
        negatives.append(f"short employment tenure ({months_emp} months)")

    failed_txn = float(applicant.get("failed_txn_ratio", 0) or 0)
    if failed_txn > 0.35:
        negatives.append(f"high failed transaction ratio ({failed_txn:.0%}) indicating cash stress")
    elif failed_txn < 0.05:
        positives.append("excellent transaction success rate")

    overdraft = int(float(applicant.get("overdraft_count", 0) or 0))
    if overdraft > 3:
        negatives.append(f"frequent overdrafts ({overdraft} occurrences)")

    neg_days = int(float(applicant.get("negative_balance_days", 0) or 0))
    if neg_days > 10:
        negatives.append(f"balance went negative for {neg_days} days in the month")

    util_pay = float(applicant.get("utility_payment_ratio", 0) or 0)
    emi_pay = float(applicant.get("emi_payment_ratio", 0) or 0)
    rent_reg = int(float(applicant.get("rent_payment_regular", 0) or 0))

    if util_pay >= 0.90 and emi_pay >= 0.90:
        positives.append("excellent utility and EMI payment discipline")
    elif util_pay < 0.50:
        negatives.append(f"poor utility payment ratio ({util_pay:.0%})")
    if emi_pay < 0.50:
        negatives.append(f"poor EMI repayment ratio ({emi_pay:.0%})")
    if rent_reg:
        positives.append("regular rent payment track record")

    emergency_savings = int(float(applicant.get("emergency_savings_months", 0) or 0))
    if emergency_savings >= 6:
        positives.append(f"strong emergency buffer ({emergency_savings} months of savings)")
    elif emergency_savings >= 3:
        positives.append(f"adequate emergency savings ({emergency_savings} months)")
    elif emergency_savings == 0:
        negatives.append("no emergency savings buffer")

    if int(float(applicant.get("property_owned", 0) or 0)):
        positives.append("owns property and has collateral support")

    fd_amount = float(applicant.get("fd_amount", 0) or 0)
    if fd_amount > 100_000:
        positives.append(f"significant fixed deposit holdings (INR {fd_amount:,.0f})")

    decision = str(oracle_result.get("decision", "")).lower()
    if decision == "deny":
        decision = "reject"

    tier = str(oracle_result.get("risk_tier") or oracle_result.get("tier") or "C")
    if tier in {"low_risk", "medium_risk", "high_risk"}:
        tier = {"low_risk": "A", "medium_risk": "B", "high_risk": "C"}[tier]
    conf = float(oracle_result.get("confidence", 0.5) or 0.5)

    positives_str = "; ".join(positives[:3]) + "." if positives else "No strong positive indicators."
    negatives_str = "; ".join(negatives[:3]) + "." if negatives else "No significant red flags."
    tier_label = {"A": "low-risk", "B": "moderate-risk", "C": "high-risk"}.get(tier, "unknown-risk")

    return (
        f"Decision: {decision.upper()} (Tier {tier} - {tier_label}, confidence {conf:.3f}). "
        f"Strengths: {positives_str} "
        f"Concerns: {negatives_str}"
    )
