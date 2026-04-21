"""
Environment interaction runner for Step 2+ work.

This preserves the old environment-style loop while moving it out of root.
It is designed to stay compatible with future OpenEnv wrapping.

Key points:
  - root `inference.py` remains the standalone FinVerse pipeline entrypoint
  - this file owns environment interaction only
  - environment actions are plain uppercase strings: APPROVE or REJECT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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

SYSTEM_PROMPT = (
    "You are a binary credit decision agent. "
    "Return exactly one token: APPROVE or REJECT."
)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


def sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def format_action(action: dict) -> str:
    return sanitize_log_value(json.dumps(action))


def profile_values(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("value", 0.0)) for field, payload in profile.items()}


def profile_confidence(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("confidence", 0.0)) for field, payload in profile.items()}


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


def policy_action(observation: Dict[str, object], task_name: str) -> str:
    del task_name
    fields = profile_values(observation)
    market_state = observation.get("market_state") or {}
    market_index = float(market_state.get("default_risk_index", 1.0))
    default_risk = _estimate_default_risk(fields, market_index)
    return "REJECT" if default_risk >= 0.50 else "APPROVE"


@dataclass
class StepLog:
    step: int
    action: str
    reward: float
    done: bool
    error: Optional[str]


class EnvironmentRunner:
    def __init__(self, env_base_url: str = ENV_BASE_URL, model_name: str = MODEL_NAME):
        self.env_base_url = env_base_url
        self.model_name = model_name

    def log_start(self, task: str) -> None:
        print(f"[START] task={task} env={BENCHMARK} model={self.model_name}", flush=True)

    def log_step(self, log: StepLog) -> None:
        error_val = log.error if log.error else "null"
        print(
            f"[STEP] step={log.step} action={log.action} reward={log.reward:.2f} "
            f"done={str(log.done).lower()} error={error_val}",
            flush=True,
        )

    def log_end(self, success: bool, steps: int, score: float, rewards: List[float]) -> None:
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )

    def call_model(self, observation: Dict[str, object], task_name: str) -> str:
        if client is None:
            return policy_action(observation, task_name)

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(observation)},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            content = (completion.choices[0].message.content or "").strip()
            action = sanitize_log_value(content).upper()
            if action not in {"APPROVE", "REJECT"}:
                raise ValueError("Model response was not APPROVE/REJECT")
            return action
        except Exception:
            return policy_action(observation, task_name)

    def run_task(self, task_name: str) -> float:
        rewards: List[float] = []
        steps_taken = 0
        observation: Dict[str, object] = {}
        final_reward = MIN_SCORE

        self.log_start(task_name)

        try:
            reset_response = requests.post(
                f"{self.env_base_url}/reset",
                json={"task_name": task_name},
                timeout=30,
            )
            reset_response.raise_for_status()
            result = reset_response.json()
            observation = result.get("observation", result)
            done = bool(result.get("done", False))

            while not done and steps_taken < MAX_STEPS:
                steps_taken += 1
                action = self.call_model(observation, task_name)
                action_str = sanitize_log_value(action)
                error = None

                try:
                    step_response = requests.post(
                        f"{self.env_base_url}/step",
                        json=action,
                        timeout=30,
                    )
                    step_response.raise_for_status()
                    result = step_response.json()
                    observation = result.get("observation", result)
                    reward = float(result.get("reward", 0.0))
                    final_reward = reward
                    done = bool(result.get("done", False))
                except Exception as exc:
                    reward = 0.0
                    final_reward = reward
                    done = True
                    error = sanitize_log_value(str(exc))

                rewards.append(reward)
                self.log_step(
                    StepLog(
                        step=steps_taken,
                        action=action_str,
                        reward=reward,
                        done=done,
                        error=error,
                    )
                )

        finally:
            final_score = strict_score(final_reward)
            self.log_end(final_score > 0.0, steps_taken, final_score, rewards)

        return final_score


def main() -> None:
    runner = EnvironmentRunner()
    tasks = [TASK_NAME] if TASK_NAME else TASKS
    for task_name in tasks:
        runner.run_task(task_name)


if __name__ == "__main__":
    main()
