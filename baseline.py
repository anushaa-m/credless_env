"""
Deterministic baseline for the FinVerse investigation workflow.
"""

import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
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


def choose_request(obs: dict) -> str | None:
    missing = obs.get("applicant", {}).get("missing_fields", [])
    for field in REQUEST_PRIORITY:
        if field in missing:
            return field
    return missing[0] if missing else None


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


def build_final_action(obs: dict, task: str) -> dict:
    fields = profile_values(obs)
    market = obs.get("market_state") or {}
    market_index = float(market.get("default_risk_index", 1.0))
    market_name = market.get("name", "Unqueried")
    risk = estimate_default_risk(fields, market_index)

    if risk < 0.35:
        tier, decision, rate = "low_risk", "approve", 10.5
    elif risk < 0.6:
        tier, decision, rate = "medium_risk", "approve", 14.0
    else:
        tier, decision, rate = "high_risk", "deny", 18.5

    rate += (market_index - 1.0) * 5.0
    reasoning = (
        f"Decision uses payment_reliability={fields.get('payment_reliability', 0.0):.2f}, "
        f"debt_burden_score={fields.get('debt_burden_score', 0.0):.2f}, "
        f"overdraft_risk={fields.get('overdraft_risk', 0.0):.2f}, and market={market_name}."
    )

    params = {"tier": tier, "rate": round(rate, 2)} if task == "risk_tiering" or decision == "approve" else {}
    return {
        "action_type": decision,
        "params": params,
        "reasoning": reasoning,
    }


def baseline_action(obs: dict, task: str) -> dict:
    target = choose_request(obs)
    if target and obs.get("step", 0) < 3:
        return {"action_type": "request_info", "params": {"field": target}, "reasoning": ""}

    if not obs.get("fraud_flags_raised") and detect_fraud(obs):
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "Low-confidence high-value fields indicate possible fabrication."},
            "reasoning": "",
        }

    if not obs.get("market_visible", False) and obs.get("step", 0) < MAX_STEPS - 1:
        return {"action_type": "query_market", "params": {}, "reasoning": ""}

    return build_final_action(obs, task)


def run_episode(task: str, retries: int = 2) -> float:
    for attempt in range(retries + 1):
        try:
            response = requests.post(f"{BASE_URL}/reset", json={"task_name": task}, timeout=30)
            response.raise_for_status()
            obs = response.json().get("observation", response.json())

            steps = 0
            while not obs.get("done") and steps < MAX_STEPS:
                steps += 1
                action = baseline_action(obs, task)
                response = requests.post(f"{BASE_URL}/step", json=action, timeout=30)
                response.raise_for_status()
                obs = response.json().get("observation", response.json())

            return strict_score(float(obs.get("episode_score", MIN_SCORE)))
        except Exception:
            if attempt == retries:
                return MIN_SCORE
            time.sleep(2 ** attempt)
    return MIN_SCORE


def main(output_json: bool = False):
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
