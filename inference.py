"""
OpenEnv hackathon inference script.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Defaults are set only for API_BASE_URL and MODEL_NAME.
The script must remain named `inference.py` at the project root and must use
the OpenAI client for all LLM calls.
"""
import json
import os
from typing import List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.environ["HF_TOKEN"]
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASKS = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
TASK_NAME = os.getenv("CREDLESS_TASK")
BENCHMARK = os.getenv("CREDLESS_BENCHMARK", "credless-env")
MAX_STEPS = 12

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are a professional credit analyst AI. "
    "Return exactly one valid JSON action object and nothing else."
)

TASK_HINTS = {
    "binary_decision": (
        'Return exactly one of: '
        '{"action_type":"approve","decision":"approve"} '
        'or {"action_type":"deny","decision":"deny"}.'
    ),
    "risk_tiering": (
        'Return exactly: '
        '{"action_type":"assign_tier","tier":"low_risk|medium_risk|high_risk","credit_limit":<float>}.'
    ),
    "adaptive_inquiry": (
        'If more information is needed, return '
        '{"action_type":"request_field","field_name":"<name>"}. '
        'Otherwise return '
        '{"action_type":"approve","decision":"approve"} '
        'or {"action_type":"deny","decision":"deny"}.'
    ),
}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


# Replace log_end function:
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def format_action(action: dict) -> str:
    return sanitize_log_value(json.dumps(action, separators=(",", ":"), sort_keys=False))


def fallback_action(task_name: str) -> dict:
    if task_name == "risk_tiering":
        return {
            "action_type": "assign_tier",
            "tier": "medium_risk",
            "credit_limit": 50000.0,
        }
    return {"action_type": "deny", "decision": "deny"}


def call_model(observation: dict, task_name: str) -> dict:
    user_prompt = (
        f"Task: {task_name}\n"
        f"Revealed fields: {json.dumps(observation.get('revealed_fields', {}), sort_keys=True)}\n"
        f"Hidden fields: {json.dumps(observation.get('hidden_fields', []))}\n"
        f"Message: {observation.get('message', '')}\n"
        f"Instruction: {TASK_HINTS.get(task_name, '')}"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=128,
            stream=False,
        )
        content = (completion.choices[0].message.content or "").strip()
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception:
        return fallback_action(task_name)


def run_task(task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    observation = {}

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
        done = bool(result.get("done", observation.get("done", False)))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = call_model(observation, task_name)
            action_str = format_action(action)
            error: Optional[str] = None

            try:
                step_response = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=30,
                )
                step_response.raise_for_status()
                result = step_response.json()
                observation = result.get("observation", result)
                reward = float(result.get("reward", observation.get("step_reward", 0.0)) or 0.0)
                done = bool(result.get("done", observation.get("done", False)))
                success = done and float(observation.get("episode_score", 0.0)) > 0.0
            except Exception as exc:
                reward = 0.0
                done = True
                error = sanitize_log_value(str(exc))
                success = False

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    finally:
        final_score = float(observation.get("episode_score", 0.0)) if observation else 0.0
        success = final_score > 0.0
        log_end(success=success, steps=steps_taken,
                score=final_score, rewards=rewards)
    return final_score


def main() -> None:
    tasks = [TASK_NAME] if TASK_NAME else TASKS
    for task_name in tasks:
        run_task(task_name)

if __name__ == "__main__":
    main()
