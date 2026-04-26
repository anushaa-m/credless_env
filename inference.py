import json
import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ✅ FIX 1: Safe env variable handling
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

# ✅ FIX 2: Correct Space URL (NOT localhost)
ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "https://anushaa-m-credless-env.hf.space"
)

TASKS = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
TASK_NAME = os.getenv("CREDLESS_TASK")
BENCHMARK = os.getenv("CREDLESS_BENCHMARK", "credless-env")
MAX_STEPS = 12
MIN_SCORE = 0.01
MAX_SCORE = 0.99

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a professional credit analyst AI. "
    "Return exactly one valid JSON action object and nothing else."
)

TASK_HINTS = {
    "binary_decision": (
        '{"action_type":"approve","decision":"approve"} OR '
        '{"action_type":"deny","decision":"deny"}'
    ),
    "risk_tiering": (
        '{"action_type":"assign_tier","tier":"low_risk|medium_risk|high_risk","credit_limit":<float>}'
    ),
    "adaptive_inquiry": (
        '{"action_type":"request_field","field_name":"<name>"} OR approve/deny'
    ),
}


# ── Logging ─────────────────────────────────────────
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


# ── Helpers ─────────────────────────────────────────
def sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def format_action(action: dict) -> str:
    return sanitize_log_value(json.dumps(action))


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


# ── Simple fallback logic (safe baseline) ───────────
def fallback_action(task_name: str) -> dict:
    if task_name == "risk_tiering":
        return {
            "action_type": "assign_tier",
            "tier": "medium_risk",
            "credit_limit": 50000.0,
        }
    return {"action_type": "deny", "decision": "deny"}


# ── Model call (safe) ───────────────────────────────
def call_model(observation: dict, task_name: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(observation)},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        content = (completion.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception:
        return fallback_action(task_name)


# ── Main loop ───────────────────────────────────────
def run_task(task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    observation = {}

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ✅ FIX 3: Safe reset call
        try:
            reset_response = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name},
                timeout=30,
            )
            reset_response.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            return MIN_SCORE

        result = reset_response.json()
        observation = result.get("observation", result)
        done = result.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

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
            steps_taken = step

            log_step(step, action_str, reward, done, error)

            if done:
                break

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
