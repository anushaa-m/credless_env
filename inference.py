import json
import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://anushaa-m-credless-env.hf.space")

TASKS = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
TASK_NAME = os.getenv("CREDLESS_TASK")
BENCHMARK = os.getenv("CREDLESS_BENCHMARK", "credless-env")
MAX_STEPS = 8
MIN_SCORE = 0.01
MAX_SCORE = 0.99

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

SYSTEM_PROMPT = (
    "You are a lending analyst in a stateful investigation environment. "
    "Return exactly one valid JSON action object and nothing else. "
    "Valid action_type values are request_info, query_market, flag_fraud, approve, deny, and escalate. "
    "Non-terminal actions can leave reasoning empty. Terminal actions must include reasoning."
)

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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def format_action(action: dict) -> str:
    return sanitize_log_value(json.dumps(action))


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


def profile_values(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("value", 0.0)) for field, payload in profile.items()}


def profile_confidence(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("confidence", 0.0)) for field, payload in profile.items()}


def _hidden_request_target(observation: Dict[str, object]) -> Optional[str]:
    hidden = observation.get("applicant", {}).get("missing_fields", [])
    for field in REQUEST_PRIORITY:
        if field in hidden:
            return field
    return hidden[0] if hidden else None


def _detect_fraud_signal(observation: Dict[str, object]) -> bool:
    fields = profile_values(observation)
    confidence = profile_confidence(observation)
    low_conf = {field: 1.0 - value for field, value in confidence.items()}
    suspicious_income = fields.get("income_capacity_score", 0.0) > 0.82 and low_conf.get("income_capacity_score", 0.0) > 0.28
    suspicious_wealth = fields.get("net_worth_score", 0.0) > 0.80 and low_conf.get("net_worth_score", 0.0) > 0.28
    suspicious_clean_profile = (
        fields.get("payment_reliability", 0.0) > 0.88
        and fields.get("overdraft_risk", 0.0) < 0.10
        and low_conf.get("payment_reliability", 0.0) > 0.25
    )
    return bool(suspicious_income or suspicious_wealth or suspicious_clean_profile)


def _market_rate_and_tier(default_risk: float, market_index: float) -> Dict[str, object]:
    if default_risk < 0.35:
        tier = "low_risk"
        rate = 10.5
    elif default_risk < 0.6:
        tier = "medium_risk"
        rate = 14.0
    else:
        tier = "high_risk"
        rate = 18.5

    rate += (market_index - 1.0) * 5.0
    decision = "approve" if tier != "high_risk" else "deny"
    return {
        "decision": decision,
        "tier": tier,
        "rate": round(rate, 2),
    }


def _estimate_default_risk(fields: Dict[str, float], market_index: float) -> float:
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


def policy_action(observation: Dict[str, object], task_name: str) -> Dict[str, object]:
    hidden = observation.get("applicant", {}).get("missing_fields", [])

    if len(hidden) > 0 and observation.get("step", 0) < 3:
        target = _hidden_request_target(observation)
        if target:
            return {"action_type": "request_info", "params": {"field": target}, "reasoning": ""}

    if not observation.get("fraud_flags_raised") and _detect_fraud_signal(observation):
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "Confidence anomalies suggest potential misreporting."},
            "reasoning": "",
        }

    if not observation.get("market_visible", False) and observation.get("step", 0) < MAX_STEPS - 1:
        return {"action_type": "query_market", "params": {}, "reasoning": ""}

    fields = profile_values(observation)
    market_state = observation.get("market_state") or {}
    market_index = float(market_state.get("default_risk_index", 1.0))
    market_name = market_state.get("name", "unqueried")
    risk = _estimate_default_risk(fields, market_index)
    pricing = _market_rate_and_tier(risk, market_index)
    reasoning = (
        f"Decision based on payment_reliability={fields.get('payment_reliability', 0.0):.2f}, "
        f"debt_burden_score={fields.get('debt_burden_score', 0.0):.2f}, "
        f"overdraft_risk={fields.get('overdraft_risk', 0.0):.2f}, and market={market_name}."
    )

    params = {"tier": pricing["tier"], "rate": pricing["rate"]} if task_name == "risk_tiering" or pricing["decision"] == "approve" else {}
    return {
        "action_type": pricing["decision"],
        "params": params,
        "reasoning": reasoning,
    }


def call_model(observation: Dict[str, object], task_name: str) -> Dict[str, object]:
    if client is None:
        return policy_action(observation, task_name)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(observation)},
            ],
            temperature=0.0,
            max_tokens=180,
        )
        content = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Model response was not a JSON object")
        return parsed
    except Exception:
        return policy_action(observation, task_name)


def run_task(task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    observation: Dict[str, object] = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_name": task_name},
            timeout=30,
        )
        reset_response.raise_for_status()
        result = reset_response.json()
        observation = result.get("observation", result)
        done = bool(result.get("done", False))

        while not done and steps_taken < MAX_STEPS:
            steps_taken += 1
            action = call_model(observation, task_name)
            action_str = format_action(action)
            error = None

            try:
                step_response = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=30,
                )
                step_response.raise_for_status()
                result = step_response.json()
                observation = result.get("observation", result)
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                success = done and float(observation.get("episode_score", 0.0)) > 0.0
            except Exception as exc:
                reward = 0.0
                done = True
                error = sanitize_log_value(str(exc))
                success = False

            rewards.append(reward)
            log_step(steps_taken, action_str, reward, done, error)

    finally:
        final_score = strict_score(float(observation.get("episode_score", MIN_SCORE)) if observation else MIN_SCORE)
        success = final_score > 0.0
        log_end(success, steps_taken, final_score, rewards)

    return final_score


def main():
    tasks = [TASK_NAME] if TASK_NAME else TASKS
    for task_name in tasks:
        run_task(task_name)


if __name__ == "__main__":
    main()
