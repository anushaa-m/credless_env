from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

from credless_model.dataset_pipeline import FEATURE_NAMES, load_dataset_cache

MODEL_PATH = Path(__file__).parent.parent / "credless_model" / "model.pkl"

MARKET_SCENARIOS: Dict[str, Dict[str, object]] = {
    "Stable Credit": {
        "risk_multiplier": 1.00,
        "threshold_delta": 0.00,
        "feature_multipliers": {},
        "summary": "Baseline credit conditions with neutral underwriting pressure.",
    },
    "Economic Boom": {
        "risk_multiplier": 0.92,
        "threshold_delta": 0.05,
        "feature_multipliers": {
            "payment_reliability": 1.04,
            "income_capacity_score": 1.05,
        },
        "summary": "Growth is strong and lenders tolerate slightly more risk.",
    },
    "High Inflation": {
        "risk_multiplier": 1.08,
        "threshold_delta": -0.03,
        "feature_multipliers": {
            "debt_burden_score": 1.08,
            "medical_stress_score": 1.05,
            "transaction_health": 0.96,
        },
        "summary": "Household cash flow is strained and missed payments rise.",
    },
    "Recession": {
        "risk_multiplier": 1.18,
        "threshold_delta": -0.06,
        "feature_multipliers": {
            "payment_reliability": 0.92,
            "income_capacity_score": 0.90,
            "employment_stability": 0.92,
            "overdraft_risk": 1.10,
            "total_delinquency_score": 1.10,
        },
        "summary": "Conservative lending climate with tighter approval requirements.",
    },
}

RAW_TIER_A_THRESHOLD = 0.65
RAW_TIER_B_THRESHOLD = 0.45


def _feature_bounds() -> Dict[str, tuple]:
    features = load_dataset_cache()["features"]
    return {
        field: (float(features[field].min()), float(features[field].max()))
        for field in FEATURE_NAMES
    }


FEATURE_BOUNDS = _feature_bounds()


def _clamp_threshold(value: float, lo: float = 0.10, hi: float = 0.90) -> float:
    return float(np.clip(value, lo, hi))


class CredLessOracle:
    def __init__(self):
        self._warned_legacy_fallback = False
        self.feature_order: List[str] = list(FEATURE_NAMES)

        if not MODEL_PATH.exists():
            print(f"[Oracle WARNING] Model not found at {MODEL_PATH}. Using fallback.")
            self.model = None
            self.model_name = "fallback"
            self.metrics = {}
            self.low_thresh = 0.40
            self.high_thresh = 0.70
            self.use_fallback = True
            return

        artifact = joblib.load(MODEL_PATH)
        self.use_fallback = False

        if isinstance(artifact, dict):
            self.model = artifact["model"]
            self.model_name = artifact.get("model_name", "unknown")
            self.metrics = artifact.get("metrics", {})
            self.feature_order = list(artifact.get("feature_names", FEATURE_NAMES))
            thresholds = artifact.get("risk_thresholds", {})
            self.low_thresh = thresholds.get("low_risk", 0.40)
            self.high_thresh = thresholds.get("medium_risk", 0.70)
        else:
            self.model = artifact
            self.model_name = "unknown"
            self.metrics = {}
            self.low_thresh = 0.40
            self.high_thresh = 0.70

        auc = self.metrics.get("test_auc", "?")
        auc_display = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        print(
            f"[Oracle] model={self.model_name} "
            f"test_auc={auc_display} "
            f"features={len(self.feature_order)} "
            f"thresholds=({self.low_thresh}, {self.high_thresh})"
        )

    def _legacy_default_prob(self, features: Dict[str, float]) -> float:
        risk_score = (
            0.16 * float(features["revolving_utilization"])
            + 0.12 * float(features["delinquency_30_59"])
            + 0.14 * float(features["delinquency_60_89"])
            + 0.18 * float(features["delinquency_90plus"])
            + 0.14 * float(features["debt_burden_score"])
            + 0.08 * float(features["medical_stress_score"])
            + 0.10 * float(features["overdraft_risk"])
            + 0.08 * float(features["location_risk_index"])
            + 0.12 * (1.0 - float(features["payment_reliability"]))
            + 0.10 * (1.0 - float(features["income_capacity_score"]))
            + 0.08 * (1.0 - float(features["employment_stability"]))
        )
        risk_score -= 0.06 * float(features["net_worth_score"])
        risk_score -= 0.05 * float(features["account_maturity"])
        return float(np.clip(risk_score, 0.0, 1.0))

    def _apply_market_adjustments(self, features: Dict[str, float], market_condition: str) -> Dict[str, float]:
        config = MARKET_SCENARIOS.get(market_condition, MARKET_SCENARIOS["Stable Credit"])
        adjusted = deepcopy(features)
        for field, multiplier in config.get("feature_multipliers", {}).items():
            if field not in adjusted:
                continue
            lo, hi = FEATURE_BOUNDS[field]
            adjusted[field] = float(np.clip(adjusted[field] * float(multiplier), lo, hi))
        return adjusted

    def predict_risk(self, features: Dict[str, float], market_condition: str = "Stable Credit") -> float:
        config = MARKET_SCENARIOS.get(market_condition, MARKET_SCENARIOS["Stable Credit"])
        adjusted_features = self._apply_market_adjustments(features, market_condition)

        if self.model is None or self.use_fallback:
            prob = self._legacy_default_prob(adjusted_features)
        else:
            x = np.array([[adjusted_features[f] for f in self.feature_order]])
            try:
                prob = float(self.model.predict_proba(x)[0][1])
            except Exception as exc:
                if not self._warned_legacy_fallback:
                    print(f"[Oracle] falling back to legacy scorer: {exc}")
                    self._warned_legacy_fallback = True
                prob = self._legacy_default_prob(adjusted_features)

        return float(np.clip(prob * float(config["risk_multiplier"]), 0.0, 1.0))

    def predict(self, features: Dict[str, float], market_condition: str = "Stable Credit") -> Dict[str, object]:
        config = MARKET_SCENARIOS.get(market_condition, MARKET_SCENARIOS["Stable Credit"])
        prob = self.predict_risk(features, market_condition=market_condition)
        low_thresh = _clamp_threshold(self.low_thresh + float(config["threshold_delta"]))
        high_thresh = _clamp_threshold(self.high_thresh + float(config["threshold_delta"]), lo=low_thresh + 0.05)

        if prob < low_thresh:
            tier, decision = "low_risk", "approve"
        elif prob < high_thresh:
            tier, decision = "medium_risk", "approve"
        else:
            tier, decision = "high_risk", "deny"

        return {
            "tier": tier,
            "decision": decision,
            "default_prob": round(prob, 6),
            "market_condition": market_condition,
            "thresholds": {
                "low_risk": round(low_thresh, 4),
                "medium_risk": round(high_thresh, 4),
            },
            "market_summary": config["summary"],
        }

    def explain_decision(self, features: Dict[str, float], market_condition: str = "Stable Credit") -> Dict[str, object]:
        risk = self.predict_risk(features, market_condition=market_condition)
        result = self.predict(features, market_condition=market_condition)
        adjusted = self._apply_market_adjustments(features, market_condition)

        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_[0], dtype=float)
            contributions = [
                (name, float(weight) * float(adjusted.get(name, 0.0)))
                for name, weight in zip(self.feature_order, coef)
            ]
        else:
            contributions = [
                ("total_delinquency_score", 0.24 * float(adjusted.get("total_delinquency_score", 0.0))),
                ("debt_burden_score", 0.18 * float(adjusted.get("debt_burden_score", 0.0))),
                ("overdraft_risk", 0.10 * float(adjusted.get("overdraft_risk", 0.0))),
                ("location_risk_index", 0.08 * float(adjusted.get("location_risk_index", 0.0))),
                ("payment_reliability", -0.16 * float(adjusted.get("payment_reliability", 0.0))),
                ("income_capacity_score", -0.14 * float(adjusted.get("income_capacity_score", 0.0))),
                ("employment_stability", -0.10 * float(adjusted.get("employment_stability", 0.0))),
                ("net_worth_score", -0.08 * float(adjusted.get("net_worth_score", 0.0))),
            ]

        top = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:3]
        dominant_feature, dominant_value = top[0] if top else ("unknown", 0.0)
        dominant_direction = "increases risk" if dominant_value >= 0 else "supports approval"
        explanation = (
            f"Decision={result['decision']} tier={result['tier']} with default_risk={risk:.4f}. "
            f"Dominant factor: {dominant_feature} ({dominant_direction})."
        )
        return {
            "decision": result["decision"],
            "tier": result["tier"],
            "default_risk": round(risk, 6),
            "feature_contributions": [(name, round(value, 6)) for name, value in top],
            "dominant_reason": dominant_feature,
            "explanation": explanation,
        }


def oracle_decision(applicant: Dict[str, object]) -> Dict[str, object]:
    """
    Deterministic rule-based oracle for raw dataset rows.

    This complements the environment's model-backed oracle instead of replacing
    it. The environment currently operates on engineered 20-feature inputs plus
    market adjustments, so `CredLessOracle.predict()` remains the primary path.
    Use this helper when a flat raw applicant row is available and a fully
    deterministic decision is preferred.
    """
    score = 0.50

    late_90 = min(float(applicant.get("numberoftimes90dayslate", 0) or 0), 15.0)
    late_60 = min(float(applicant.get("numberoftime60-89dayspastduenotworse", 0) or 0), 15.0)
    late_30 = min(float(applicant.get("numberoftime30-59dayspastduenotworse", 0) or 0), 15.0)
    credit_penalty = (late_90 * 0.060) + (late_60 * 0.030) + (late_30 * 0.012)
    score -= min(credit_penalty, 0.35)

    debt_ratio = min(float(applicant.get("debtratio", 0.35) or 0.35), 2.0)
    if debt_ratio > 0.40:
        score -= min((debt_ratio - 0.40) * 0.25, 0.25)
    elif debt_ratio < 0.20:
        score += 0.05

    rev_util = min(float(applicant.get("revolvingutilizationofunsecuredlines", 0.30) or 0.30), 1.0)
    if rev_util > 0.60:
        score -= (rev_util - 0.60) * 0.20
    elif rev_util < 0.20:
        score += 0.04

    income = float(applicant.get("monthlyincome", 0) or 0)
    if income >= 50_000:
        score += 0.12
    elif income >= 20_000:
        score += 0.08
    elif income >= 10_000:
        score += 0.04
    elif income >= 5_000:
        score += 0.01
    elif income < 2_000:
        score -= 0.10

    variability = float(applicant.get("income_variability_score", 0.3) or 0.3)
    score -= variability * 0.08

    months_employed = float(applicant.get("months_employed", 0) or 0)
    if months_employed >= 36:
        score += 0.06
    elif months_employed >= 12:
        score += 0.03
    elif months_employed < 3:
        score -= 0.06

    emp_type = str(applicant.get("employment_type", "") or "").lower()
    if emp_type == "salaried":
        score += 0.04
    elif emp_type == "unemployed":
        score -= 0.10
    elif emp_type == "self_employed":
        monthly_revenue = float(applicant.get("monthly_revenue", 0) or 0)
        profit_margin = float(applicant.get("profit_margin", 0) or 0)
        if monthly_revenue > 50_000 and profit_margin > 0.20:
            score += 0.05
        business_age = float(applicant.get("business_age", 0) or 0)
        if business_age >= 36:
            score += 0.03

    failed_txn = float(applicant.get("failed_txn_ratio", 0.1) or 0.1)
    if failed_txn > 0.35:
        score -= (failed_txn - 0.35) * 0.30
    elif failed_txn < 0.05:
        score += 0.03

    late_night = float(applicant.get("late_night_txn_ratio", 0.1) or 0.1)
    if late_night > 0.40:
        score -= (late_night - 0.40) * 0.10

    active_days = float(applicant.get("active_txn_days", 15) or 15)
    if active_days >= 20:
        score += 0.03
    elif active_days < 5:
        score -= 0.03

    overdraft = float(applicant.get("overdraft_count", 0) or 0)
    score -= min(overdraft * 0.02, 0.08)

    neg_bal_days = float(applicant.get("negative_balance_days", 0) or 0)
    score -= min(neg_bal_days * 0.004, 0.08)

    util_pay = float(applicant.get("utility_payment_ratio", 0.5) or 0.5)
    score += (util_pay - 0.50) * 0.12

    emi_pay = float(applicant.get("emi_payment_ratio", 0.5) or 0.5)
    score += (emi_pay - 0.50) * 0.12

    rent_reg = int(applicant.get("rent_payment_regular", 0) or 0)
    if rent_reg == 1:
        score += 0.04

    em_savings = float(applicant.get("emergency_savings_months", 0) or 0)
    if em_savings >= 6:
        score += 0.08
    elif em_savings >= 3:
        score += 0.04
    elif em_savings == 0:
        score -= 0.03

    fd_amount = float(applicant.get("fd_amount", 0) or 0)
    if fd_amount >= 100_000:
        score += 0.06
    elif fd_amount >= 25_000:
        score += 0.03

    avg_balance = float(applicant.get("avg_monthly_balance", 0) or 0)
    if avg_balance >= 50_000:
        score += 0.04
    elif avg_balance < 1_000:
        score -= 0.03

    if int(applicant.get("property_owned", 0) or 0):
        score += 0.05
    if int(applicant.get("vehicle_owned", 0) or 0):
        score += 0.02

    gold_value = float(applicant.get("gold_value_estimate", 0) or 0)
    if gold_value >= 50_000:
        score += 0.03

    if int(applicant.get("current_medical_condition", 0) or 0):
        medical_debt = float(applicant.get("medical_debt", 0) or 0)
        score -= 0.06 if medical_debt > 50_000 else 0.02

    location_risk = float(applicant.get("location_risk_index", 0.3) or 0.3)
    score -= location_risk * 0.06

    age = float(applicant.get("age", 35) or 35)
    if 25 <= age <= 55:
        score += 0.01
    elif age < 21 or age > 70:
        score -= 0.02

    dependents = float(applicant.get("numberofdependents", 0) or 0)
    if dependents > 3:
        score -= 0.02

    score = round(float(np.clip(score, 0.0, 1.0)), 4)

    if score >= RAW_TIER_A_THRESHOLD:
        decision = "approve"
        tier = "low_risk"
        risk_tier = "A"
    elif score >= RAW_TIER_B_THRESHOLD:
        decision = "approve"
        tier = "medium_risk"
        risk_tier = "B"
    else:
        decision = "deny"
        tier = "high_risk"
        risk_tier = "C"

    return {
        "decision": decision,
        "tier": tier,
        "risk_tier": risk_tier,
        "confidence": score,
        "default_prob": round(1.0 - score, 4),
    }
