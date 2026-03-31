# inference.py  ← MUST be named exactly this, at project root
"""
OpenEnv Hackathon — required inference script.
Uses OpenAI client pointed at HF router.

Required env vars:
    API_BASE_URL  — e.g. https://router.huggingface.co/v1
    MODEL_NAME    — e.g. meta-llama/Llama-3.1-8B-Instruct
    HF_TOKEN      — your Hugging Face access token
    ENV_BASE_URL  — running environment URL (default http://localhost:7860)
"""
import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

# ── Required env vars ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ["HF_TOKEN"]          # hard fail if missing
ENV_BASE_URL = os.getenv("ENV_BASE_URL",  "http://localhost:7860")
RUNS         = int(os.getenv("INFERENCE_RUNS", "5"))

# ── OpenAI client pointed at HF router ────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = (
    "You are a professional credit analyst AI. "
    "Evaluate loan applicants using behavioural financial data. "
    "Return ONLY a single valid JSON action object — no explanation, no markdown."
)

TASK_HINTS = {
    "binary_decision": (
        'Return exactly one of:\n'
        '  {"action_type":"approve","decision":"approve"}\n'
        '  {"action_type":"deny","decision":"deny"}'
    ),
    "risk_tiering": (
        'Return exactly:\n'
        '  {"action_type":"assign_tier",'
        '"tier":"low_risk|medium_risk|high_risk",'
        '"credit_limit":<float in INR>}'
    ),
    "adaptive_inquiry": (
        'You may first request hidden fields (up to 3 free):\n'
        '  {"action_type":"request_field","field_name":"<name>"}\n'
        'Then make final decision:\n'
        '  {"action_type":"approve","decision":"approve|deny"}'
    ),
}


def _call_model(obs: dict) -> dict:
    """Call the LLM and parse JSON action."""
    task = obs.get("task_name", "binary_decision")
    user = (
        f"Task: {task}\n\n"
        f"Applicant profile:\n{json.dumps(obs.get('revealed_fields', {}), indent=2)}\n\n"
        f"Hidden fields available: {obs.get('hidden_fields', [])}\n\n"
        f"Environment message: {obs.get('message', '')}\n\n"
        f"Required format:\n{TASK_HINTS.get(task, '')}\n\n"
        "Your JSON action:"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user},
            ],
            max_tokens=128,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback safe action
        return {"action_type": "deny", "decision": "deny"}
    except Exception:
        return {"action_type": "deny", "decision": "deny"}


def run_episode(task_name: str, retries: int = 2) -> float:
    """Run one full episode and return final episode_score."""
    for attempt in range(retries + 1):
        try:
            # Reset
            r   = requests.post(
                f"{ENV_BASE_URL}/reset",
                json={"task_name": task_name},
                timeout=30,
            )
            r.raise_for_status()
            obs = r.json().get("observation", r.json())

            # Episode loop
            for _ in range(12):
                if obs.get("done"):
                    break
                action = _call_model(obs)
                r      = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json=action,
                    timeout=30,
                )
                r.raise_for_status()
                obs = r.json().get("observation", r.json())

            return float(obs.get("episode_score", 0.0))

        except Exception as e:
            if attempt == retries:
                return 0.0
            time.sleep(2 ** attempt)   # exponential backoff

    return 0.0


def main(output_json: bool = False) -> dict:
    tasks   = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
    results = {}

    for task in tasks:
        scores = []
        for i in range(RUNS):
            score = run_episode(task)
            scores.append(round(score, 4))
            if not output_json:
                bar = "█" * int(score * 20)
                print(f"  [{i+1}/{RUNS}] {task:25s} [{bar:<20}] {score:.3f}")

        mean = round(sum(scores) / len(scores), 4)
        results[task] = {
            "mean_score": mean,
            "scores":     scores,
            "runs":       RUNS,
            "model":      MODEL_NAME,
        }

    if output_json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=== CredLess-Env Inference Results ===")
        for task, r in results.items():
            bar = "█" * int(r["mean_score"] * 20)
            print(f"  {task:25s} [{bar:<20}] mean={r['mean_score']:.3f}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CredLess-Env inference script")
    p.add_argument("--output-json", action="store_true",
                   help="Print JSON to stdout (used by /baseline endpoint)")
    args = p.parse_args()
    main(args.output_json)