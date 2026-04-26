"""
Deterministic baseline for the binary CredLess decision environment.
"""

import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
FALLBACK_BASE_URL = "http://localhost:7860"
RUNS = int(os.getenv("BASELINE_RUNS", "5"))
MIN_SCORE = 0.01
MAX_SCORE = 0.99
MAX_STEPS = 8
REQUEST_PRIORITY = [
    "total_delinquency_score",
    "overdraft_risk",
    "medical_stress_score",
    "debt_burden_score",
    "payment_reliability",
    "income_capacity_score",
    "employment_stability",
    "account_maturity",
]


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


def resolve_base_url() -> str:
    candidates = [str(BASE_URL).rstrip("/"), FALLBACK_BASE_URL]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            response = requests.get(f"{candidate}/health", timeout=5)
            response.raise_for_status()
            return candidate
        except Exception:
            continue
    return str(BASE_URL).rstrip("/")


def oracle_risk(payload: dict) -> float:
    return float(payload.get("oracle_risk", 0.0) or 0.0)


def oracle_confidence(payload: dict) -> float:
    return float(payload.get("oracle_confidence", 0.0) or 0.0)


def oracle_factor_names(payload: dict) -> list[str]:
    top_factors = payload.get("top_factors", []) or []
    names: list[str] = []
    for item in top_factors:
        if isinstance(item, list) and item:
            names.append(str(item[0]))
    return names


def profile_values(obs: dict) -> dict:
    profile = obs.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("value", 0.0)) for field, payload in profile.items()}


def profile_confidence(obs: dict) -> dict:
    profile = obs.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("confidence", 0.0)) for field, payload in profile.items()}


def detect_fraud(obs: dict) -> bool:
    fields = profile_values(obs)
    confidence = profile_confidence(obs)
    low_confidence = {field: 1.0 - value for field, value in confidence.items()}
    return bool(
        fields.get("income_capacity_score", 0.0) > 0.82 and low_confidence.get("income_capacity_score", 0.0) > 0.28
        or fields.get("net_worth_score", 0.0) > 0.80 and low_confidence.get("net_worth_score", 0.0) > 0.28
        or (
            fields.get("payment_reliability", 0.0) > 0.88
            and fields.get("overdraft_risk", 0.0) < 0.10
            and low_confidence.get("payment_reliability", 0.0) > 0.25
        )
    )


def estimate_default_risk(fields: dict, market_index: float) -> float:
    risk = (
        0.16 * fields.get("revolving_utilization", 0.5)
        + 0.08 * fields.get("delinquency_30_59", 0.0)
        + 0.10 * fields.get("delinquency_60_89", 0.0)
        + 0.12 * fields.get("delinquency_90plus", 0.0)
        + 0.12 * fields.get("total_delinquency_score", 0.0)
        + 0.12 * fields.get("debt_burden_score", 0.0)
        + 0.08 * fields.get("medical_stress_score", 0.0)
        + 0.08 * fields.get("overdraft_risk", 0.0)
        + 0.06 * fields.get("location_risk_index", 0.0)
        + 0.08 * (1.0 - fields.get("payment_reliability", 0.5))
        + 0.07 * (1.0 - fields.get("income_capacity_score", 0.5))
        + 0.05 * (1.0 - fields.get("employment_stability", 0.5))
        + 0.04 * (1.0 - fields.get("account_maturity", 0.5))
    )
    risk -= 0.04 * fields.get("net_worth_score", 0.5)
    risk -= 0.02 * fields.get("asset_ownership_score", 0.5)
    risk *= market_index
    return max(0.0, min(1.0, risk))


def build_terminal_action(obs: dict, payload: dict, task: str) -> dict:
    fields = profile_values(obs)
    market = obs.get("market_state") or {}
    market_index = float(market.get("default_risk_index", 1.0))
    heuristic_risk = estimate_default_risk(fields, market_index)
    model_risk = oracle_risk(payload)
    confidence = oracle_confidence(payload)
    risk = 0.60 * model_risk + 0.40 * heuristic_risk if model_risk > 0.0 else heuristic_risk

    if task == "adaptive_inquiry" and confidence < 0.60:
        return {
            "action_type": "escalate",
            "reasoning": "Oracle confidence is low after investigation; escalating for manual review.",
        }
    if risk < 0.35:
        decision = "approve"
    elif risk < 0.6:
        decision = "approve"
    else:
        decision = "deny"

    return {
        "action_type": decision,
        "reasoning": (
            f"Decision based on oracle_risk={model_risk:.3f}, "
            f"oracle_confidence={confidence:.3f}, and revealed repayment signals."
        ),
    }


def baseline_action(payload: dict, task: str) -> dict:
    obs = payload.get("observation", payload)
    profile = obs.get("applicant", {}).get("profile", {})
    missing_fields = list(obs.get("applicant", {}).get("missing_fields", []))
    required_fields = list((obs.get("current_policy") or {}).get("required_fields", []))
    factor_names = oracle_factor_names(payload)

    for field in required_fields:
        if field in missing_fields:
            return {"action_type": "request_info", "params": {"field": field}}

    for field in factor_names:
        if field in missing_fields and field not in profile:
            return {"action_type": "request_info", "params": {"field": field}}

    for field in REQUEST_PRIORITY:
        if field in missing_fields and field not in profile:
            return {"action_type": "request_info", "params": {"field": field}}

    if not obs.get("market_visible", False) and oracle_confidence(payload) < 0.85:
        return {"action_type": "query_market"}

    if (detect_fraud(obs) or oracle_risk(payload) > 0.72) and not obs.get("fraud_flags_raised"):
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "transaction inconsistency, elevated oracle risk, or low-confidence profile"},
            "reasoning": "Potential fraud indicators detected from observed profile and oracle risk signal.",
        }

    return build_terminal_action(obs, payload, task)


def run_episode(task: str, retries: int = 2) -> float:
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{BASE_URL}/reset", json={"task_name": task}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            session_id = str(payload["session_id"])
            episode_id = str(payload["episode_id"])
            obs = payload.get("observation", payload)

            steps = 0
            done = bool(payload.get("done", False))
            final_reward = MIN_SCORE
            while not done and steps < MAX_STEPS:
                steps += 1
                action = baseline_action(payload, task)
                response = requests.post(
                    f"{BASE_URL}/step",
                    json={
                        "session_id": session_id,
                        "episode_id": episode_id,
                        "action": action,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                obs = payload.get("observation", obs)
                done = bool(payload.get("done", False))
                final_reward = float(payload.get("reward", MIN_SCORE))
            return strict_score(final_reward)
        except Exception:
            if attempt == retries:
                return MIN_SCORE
            time.sleep(2 ** attempt)
    return MIN_SCORE


def main(output_json: bool = False):
    global BASE_URL
    BASE_URL = resolve_base_url()
    tasks = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
    results = {}
    for task in tasks:
        scores = [run_episode(task) for _ in range(RUNS)]
        results[task] = {
            "mean_score": strict_score(sum(scores) / len(scores)),
            "scores": [strict_score(s) for s in scores],
            "runs": RUNS,
        }

    if output_json:
        print(json.dumps(results))
    else:
        print("\n=== FinVerse Baseline ===")
        for task, result in results.items():
            bar = "#" * int(result["mean_score"] * 20)
            print(f"  {task:25s} [{bar:<20}] {result['mean_score']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true")
    main(parser.parse_args().output_json)
