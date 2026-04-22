from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from credless_model.dataset_pipeline import FEATURE_NAMES
from pipeline.oracle import oracle_decision
from pipeline.reasoning import generate_reasoning


ROOT = Path(__file__).resolve().parent.parent
AGENT2_MODULE_PATH = ROOT / "agent2-decision-base" / "train.py"
FINVERSE_MODEL_PATH = ROOT / "models" / "saved" / "finverse_model.pkl"
FINVERSE_SCALER_PATH = ROOT / "models" / "saved" / "scaler.pkl"
LEGACY_AGENT1_PATH = ROOT / "credless_model" / "model.pkl"

MARITAL_STATUS_MAP = {
    "single": 0.0,
    "married": 1.0,
    "divorced": 2.0,
    "widowed": 3.0,
    "unknown": 4.0,
}
EMPLOYMENT_TYPE_MAP = {
    "unemployed": 0.0,
    "contract": 1.0,
    "self_employed": 2.0,
    "salaried": 3.0,
    "retired": 4.0,
    "unknown": 5.0,
}
RAW_DEFAULTS = {
    "age": 35.0,
    "monthlyincome": 25000.0,
    "debtratio": 0.35,
    "revolvingutilizationofunsecuredlines": 0.30,
    "numberoftime30-59dayspastduenotworse": 0.0,
    "numberoftime60-89dayspastduenotworse": 0.0,
    "numberoftimes90dayslate": 0.0,
    "numberofopencreditlinesandloans": 6.0,
    "numberrealestateloansorlines": 1.0,
    "years_at_address": 5.0,
    "months_employed": 36.0,
    "monthly_revenue": 0.0,
    "profit_margin": 0.15,
    "business_age": 0.0,
    "business_type_risk": 0.35,
    "debtratio": 0.35,
    "medical_debt": 0.0,
    "current_medical_condition": 0.0,
    "emi_payment_ratio": 0.85,
    "rent_payment_regular": 1.0,
    "utility_payment_ratio": 0.90,
    "fd_amount": 0.0,
    "gold_value_estimate": 0.0,
    "property_owned": 0.0,
    "vehicle_owned": 0.0,
    "emergency_savings_months": 2.0,
    "bank_account_age_months": 48.0,
    "avg_monthly_balance": 15000.0,
    "negative_balance_days": 0.0,
    "overdraft_count": 0.0,
    "salary_credit_consistency": 0.85,
    "income_variability_score": 0.30,
    "monthly_upi_spend": 8000.0,
    "active_txn_days": 18.0,
    "failed_txn_ratio": 0.08,
    "late_night_txn_ratio": 0.08,
    "gov_scheme_enrollment": 0.0,
    "location_risk_index": 0.35,
    "numberofdependents": 1.0,
}


def _load_agent2_module():
    module_name = "credless_agent2_module"
    spec = importlib.util.spec_from_file_location(module_name, AGENT2_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Agent 2 module from {AGENT2_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_user_data(user_data: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(user_data).items():
        if isinstance(value, np.generic):
            value = value.item()
        normalized[str(key).strip().lower().replace(" ", "_")] = value
    return normalized


class FrozenRiskPredictor:
    def __init__(self) -> None:
        self.raw_feature_defaults = dict(RAW_DEFAULTS)
        self._feature_stats = self._compute_feature_stats()

        self.backend = ""
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.feature_order: list[str] = []
        self.positive_class_meaning = "default_risk"

        if FINVERSE_MODEL_PATH.exists() and FINVERSE_SCALER_PATH.exists():
            with open(FINVERSE_MODEL_PATH, "rb") as handle:
                artifact = pickle.load(handle)
            with open(FINVERSE_SCALER_PATH, "rb") as handle:
                self.scaler = pickle.load(handle)
            self.model = artifact["model"]
            self.feature_order = list(artifact["feature_cols"])
            self.backend = "finverse_saved"
            self.positive_class_meaning = "approve"
            return

        if LEGACY_AGENT1_PATH.exists():
            artifact = joblib.load(LEGACY_AGENT1_PATH)
            self.model = artifact["model"] if isinstance(artifact, dict) else artifact
            self.feature_order = list(artifact.get("feature_names", FEATURE_NAMES)) if isinstance(artifact, dict) else list(FEATURE_NAMES)
            self.backend = "credless_legacy"
            self.positive_class_meaning = "default_risk"
            return

        raise FileNotFoundError(
            "No supported Agent 1 artifact was found. Expected either "
            f"{FINVERSE_MODEL_PATH} and {FINVERSE_SCALER_PATH}, or {LEGACY_AGENT1_PATH}."
        )

    def predict(self, user_data: Mapping[str, Any]) -> float:
        if self.backend == "finverse_saved":
            approve_probability, _ = self._predict_finverse(user_data)
            return float(np.clip(1.0 - approve_probability, 0.0, 1.0))
        default_probability, _ = self._predict_legacy(user_data)
        return float(np.clip(default_probability, 0.0, 1.0))

    def explain(self, user_data: Mapping[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
        if self.backend == "finverse_saved":
            _, details = self._predict_finverse(user_data)
        else:
            _, details = self._predict_legacy(user_data)

        contributions = details.get("contributions", [])
        top_items = sorted(contributions, key=lambda item: abs(float(item[1])), reverse=True)[:top_k]
        explanation: list[dict[str, Any]] = []
        for name, contribution in top_items:
            numeric_value = float(contribution)
            explanation.append(
                {
                    "feature": str(name),
                    "contribution": round(numeric_value, 6),
                    "impact": "raises risk" if numeric_value >= 0 else "reduces risk",
                }
            )
        if not explanation:
            risk_score = self.predict(user_data)
            explanation.append(
                {
                    "feature": "risk_score",
                    "contribution": round(risk_score - 0.50, 6),
                    "impact": "raises risk" if risk_score >= 0.50 else "reduces risk",
                }
            )
        return explanation

    def _predict_finverse(self, user_data: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        vector_frame = self._prepare_finverse_frame(user_data)
        scaled = self.scaler.transform(vector_frame[self.feature_order].to_numpy())
        probability = float(self.model.predict_proba(scaled)[0][1])

        contributions: list[tuple[str, float]] = []
        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_[0], dtype=float)
            risk_contrib = -coef * scaled[0]
            contributions = list(zip(self.feature_order, risk_contrib.tolist()))
        elif hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_, dtype=float)
            row = vector_frame[self.feature_order].iloc[0]
            contributions = []
            for feature_name, importance in zip(self.feature_order, importances):
                baseline = self._feature_stats.get(feature_name, {"mean": 0.0, "std": 1.0})
                z_score = (float(row[feature_name]) - baseline["mean"]) / baseline["std"]
                contributions.append((feature_name, float(z_score * importance)))

        return probability, {"contributions": contributions}

    def _predict_legacy(self, user_data: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
        feature_frame = self._prepare_legacy_features(user_data)
        vector = feature_frame[self.feature_order].to_numpy()
        probability = float(self.model.predict_proba(vector)[0][1])

        contributions: list[tuple[str, float]] = []
        if hasattr(self.model, "coef_"):
            coef = np.asarray(self.model.coef_[0], dtype=float)
            contributions = list(zip(self.feature_order, (coef * vector[0]).tolist()))

        return probability, {"contributions": contributions}

    def _prepare_finverse_frame(self, user_data: Mapping[str, Any]) -> pd.DataFrame:
        values = _normalize_user_data(user_data)
        record: dict[str, float] = {}

        for feature_name in self.feature_order:
            if feature_name.endswith("_enc"):
                base = feature_name[:-4]
                raw_value = str(values.get(base, "unknown") or "unknown").strip().lower()
                if base == "marital_status":
                    record[feature_name] = MARITAL_STATUS_MAP.get(raw_value, MARITAL_STATUS_MAP["unknown"])
                elif base == "employment_type":
                    record[feature_name] = EMPLOYMENT_TYPE_MAP.get(raw_value, EMPLOYMENT_TYPE_MAP["unknown"])
                else:
                    record[feature_name] = 0.0
                continue

            raw_value = values.get(feature_name, self.raw_feature_defaults.get(feature_name, 0.0))
            try:
                record[feature_name] = float(raw_value)
            except (TypeError, ValueError):
                record[feature_name] = float(self.raw_feature_defaults.get(feature_name, 0.0))

        return pd.DataFrame([record], columns=self.feature_order)

    def _prepare_legacy_features(self, user_data: Mapping[str, Any]) -> pd.DataFrame:
        values = _normalize_user_data(user_data)
        row_features = self._engineer_legacy_row(values)
        return pd.DataFrame([row_features], columns=self.feature_order)

    def _compute_feature_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for column, default in RAW_DEFAULTS.items():
            scale = max(abs(float(default)), 1.0)
            stats[column] = {"mean": float(default), "std": scale}
        return stats

    def _engineer_legacy_row(self, values: Mapping[str, Any]) -> dict[str, float]:
        def _num(name: str) -> float:
            raw = values.get(name, self.raw_feature_defaults.get(name, 0.0))
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(self.raw_feature_defaults.get(name, 0.0))

        def _clip01(name: str, upper: float) -> float:
            return float(np.clip(_num(name) / upper, 0.0, 1.0))

        def _raw_clip01(name: str) -> float:
            return float(np.clip(_num(name), 0.0, 1.0))

        def _log_scale(name: str, upper: float) -> float:
            return float(np.clip(np.log1p(max(_num(name), 0.0)) / np.log1p(upper), 0.0, 1.0))

        delinquency_30 = _clip01("numberoftime30-59dayspastduenotworse", 10.0)
        delinquency_60 = _clip01("numberoftime60-89dayspastduenotworse", 10.0)
        delinquency_90 = _clip01("numberoftimes90dayslate", 10.0)
        debt_ratio = _clip01("debtratio", 5.0)
        real_estate = _clip01("numberrealestateloansorlines", 10.0)
        medical_debt = _log_scale("medical_debt", 250000.0)
        medical_condition = _raw_clip01("current_medical_condition")
        emi_ratio = _raw_clip01("emi_payment_ratio")
        income_capacity = float(
            np.clip(
                0.45 * _log_scale("monthlyincome", 200000.0)
                + 0.30 * _log_scale("monthly_revenue", 500000.0)
                + 0.15 * _raw_clip01("profit_margin")
                + 0.10 * _clip01("emergency_savings_months", 24.0),
                0.0,
                1.0,
            )
        )
        net_worth = float(
            np.clip(
                0.40 * _log_scale("fd_amount", 500000.0)
                + 0.30 * _log_scale("gold_value_estimate", 250000.0)
                + 0.30 * _clip01("emergency_savings_months", 24.0),
                0.0,
                1.0,
            )
        )
        asset_ownership = float(
            np.clip(
                0.60 * _raw_clip01("property_owned") + 0.40 * _raw_clip01("vehicle_owned"),
                0.0,
                1.0,
            )
        )
        payment_reliability = float(
            np.clip(
                0.28 * _raw_clip01("utility_payment_ratio")
                + 0.25 * _raw_clip01("emi_payment_ratio")
                + 0.22 * _raw_clip01("salary_credit_consistency")
                + 0.15 * _raw_clip01("rent_payment_regular")
                + 0.10 * (1.0 - _clip01("negative_balance_days", 30.0)),
                0.0,
                1.0,
            )
        )
        transaction_health = float(
            np.clip(
                0.35 * _clip01("active_txn_days", 31.0)
                + 0.30 * _log_scale("monthly_upi_spend", 100000.0)
                + 0.20 * (1.0 - _clip01("failed_txn_ratio", 0.5))
                + 0.15 * (1.0 - _raw_clip01("late_night_txn_ratio")),
                0.0,
                1.0,
            )
        )
        employment_stability = float(
            np.clip(
                0.35 * _log_scale("months_employed", 360.0)
                + 0.30 * min(EMPLOYMENT_TYPE_MAP.get(str(values.get("employment_type", "unknown")).strip().lower(), 0.0) / 5.0, 1.0)
                + 0.20 * _clip01("years_at_address", 30.0)
                + 0.15 * (1.0 - _raw_clip01("income_variability_score")),
                0.0,
                1.0,
            )
        )
        account_maturity = float(
            np.clip(
                0.55 * _clip01("bank_account_age_months", 120.0)
                + 0.25 * _log_scale("avg_monthly_balance", 500000.0)
                + 0.20 * _clip01("years_at_address", 30.0),
                0.0,
                1.0,
            )
        )
        business_health = float(
            np.clip(
                0.40 * _clip01("business_age", 30.0)
                + 0.35 * (1.0 - _raw_clip01("business_type_risk"))
                + 0.25 * _raw_clip01("gov_scheme_enrollment"),
                0.0,
                1.0,
            )
        )

        features = {
            "revolving_utilization": _raw_clip01("revolvingutilizationofunsecuredlines"),
            "delinquency_30_59": delinquency_30,
            "delinquency_60_89": delinquency_60,
            "delinquency_90plus": delinquency_90,
            "total_delinquency_score": float(np.clip(0.20 * delinquency_30 + 0.30 * delinquency_60 + 0.50 * delinquency_90, 0.0, 1.0)),
            "debt_burden_score": float(np.clip(0.35 * debt_ratio + 0.25 * emi_ratio + 0.20 * medical_debt + 0.10 * medical_condition + 0.10 * real_estate, 0.0, 1.0)),
            "real_estate_exposure": real_estate,
            "medical_stress_score": float(np.clip(0.60 * medical_debt + 0.40 * medical_condition, 0.0, 1.0)),
            "income_capacity_score": income_capacity,
            "net_worth_score": net_worth,
            "asset_ownership_score": asset_ownership,
            "payment_reliability": payment_reliability,
            "transaction_health": transaction_health,
            "overdraft_risk": _clip01("overdraft_count", 20.0),
            "employment_stability": employment_stability,
            "account_maturity": account_maturity,
            "business_health_score": business_health,
            "age_years": _clip01("age", 90.0),
            "open_credit_lines": _clip01("numberofopencreditlinesandloans", 30.0),
            "location_risk_index": _raw_clip01("location_risk_index"),
        }
        return {feature_name: float(features.get(feature_name, 0.0)) for feature_name in self.feature_order}


class CreditDecisionEnvironment:
    def __init__(self, user_data: Mapping[str, Any]) -> None:
        self.user_data = _normalize_user_data(user_data)
        self.oracle = oracle_decision(self.user_data)
        self.done = False

    def step(self, action: str) -> dict[str, Any]:
        normalized_action = "APPROVE" if str(action).upper() == "APPROVE" else "REJECT"
        approve_confidence = float(self.oracle["confidence"])
        reward = approve_confidence if normalized_action == "APPROVE" else 1.0 - approve_confidence
        explanation = generate_reasoning(self.user_data, self.oracle)
        self.done = True
        return {
            "reward": float(reward),
            "done": True,
            "info": {
                "explanation": explanation,
                "oracle_score": float(reward),
                "oracle_decision": "APPROVE" if self.oracle["decision"] == "approve" else "REJECT",
                "oracle_confidence": approve_confidence,
            },
        }


class CreditDecisionPipeline:
    def __init__(
        self,
        *,
        agent1: FrozenRiskPredictor | None = None,
        agent2: Any | None = None,
    ) -> None:
        self.agent1 = agent1 or FrozenRiskPredictor()
        if agent2 is None:
            module = _load_agent2_module()
            agent2 = module.Agent2DecisionMaker.from_checkpoint()
        self.agent2 = agent2

    def run(self, user_data: Mapping[str, Any], *, shap_info: Sequence[Mapping[str, Any]] | None = None, env: Any | None = None) -> dict[str, Any]:
        normalized_user = _normalize_user_data(user_data)
        risk_score = float(self.agent1.predict(normalized_user))
        active_shap = list(shap_info) if shap_info is not None else self.agent1.explain(normalized_user)
        policy_output = self.agent2.generate_with_metadata(normalized_user, risk_score, active_shap)
        active_env = env or CreditDecisionEnvironment(normalized_user)
        raw_result = active_env.step(policy_output.decision)
        result = _normalize_env_result(raw_result, active_env)
        result.update(
            {
                "user_data": normalized_user,
                "risk_score": risk_score,
                "shap_info": active_shap,
                "decision": policy_output.decision,
                "policy_output": {
                    "raw_text": policy_output.raw_text,
                    "prompt": policy_output.prompt,
                    "logprob": float(policy_output.logprob),
                    "approve_probability": float(policy_output.approve_probability),
                },
            }
        )
        return result


def _normalize_env_result(raw_result: Any, env: Any) -> dict[str, Any]:
    if isinstance(raw_result, dict):
        return {
            "reward": float(raw_result.get("reward", 0.0)),
            "done": bool(raw_result.get("done", False)),
            "info": dict(raw_result.get("info", {})),
        }

    info = {}
    if hasattr(env, "last_info"):
        maybe_info = env.last_info()
        if isinstance(maybe_info, dict):
            info = maybe_info

    reward = float(getattr(raw_result, "step_reward", 0.0))
    done = bool(getattr(raw_result, "done", False))
    if "oracle_score" not in info:
        info["oracle_score"] = reward
    if "explanation" not in info:
        info["explanation"] = getattr(raw_result, "message", "")
    return {"reward": reward, "done": done, "info": info}


_PIPELINE_SINGLETON: CreditDecisionPipeline | None = None


def run_pipeline(user_data: Mapping[str, Any]) -> dict[str, Any]:
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        _PIPELINE_SINGLETON = CreditDecisionPipeline()

    risk = _PIPELINE_SINGLETON.agent1.predict(user_data)
    decision = _PIPELINE_SINGLETON.agent2.generate_decision(
        user_data,
        risk,
        shap_info=None,
    )
    env = CreditDecisionEnvironment(user_data)
    result = env.step(decision)
    normalized = _normalize_env_result(result, env)
    normalized.update(
        {
            "risk_score": float(risk),
            "decision": str(decision),
            "user_data": _normalize_user_data(user_data),
        }
    )
    return normalized
