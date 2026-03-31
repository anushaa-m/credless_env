# baseline.py
"""
Legacy baseline script (kept for compatibility).
For hackathon submission, the main script is inference.py.

Reads: OPENAI_API_KEY (required), ENV_BASE_URL (optional)
"""
import os
import json
import time
import argparse
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MODEL    = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
RUNS     = int(os.getenv("BASELINE_RUNS", "5"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM = (
    "You are a credit analyst AI. "
    "Return ONLY a valid JSON action object — no extra text."
)

HINTS = {
    "binary_decision":  'Return {"action_type":"approve","decision":"approve"} or {"action_type":"deny","decision":"deny"}',
    "risk_tiering":     'Return {"action_type":"assign_tier","tier":"low_risk|medium_risk|high_risk","credit_limit":<INR float>}',
    "adaptive_inquiry": 'First request fields: {"action_type":"request_field","field_name":"<name>"}, then decide.',
}


def _call_llm(obs: dict) -> dict:         # ✅ FIXED: was named _llm in run_episode call
    task = obs.get("task_name", "binary_decision")
    user = (
        f"Task: {task}\n"
        f"Profile: {json.dumps(obs.get('revealed_fields', {}), indent=2)}\n"
        f"Hidden fields: {obs.get('hidden_fields', [])}\n"
        f"Message: {obs.get('message', '')}\n\n"
        f"Format hint: {HINTS.get(task, '')}\n"
        "Reply with ONE JSON action object only."
    )
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(r.choices[0].message.content)
    except Exception:
        return {"action_type": "deny", "decision": "deny"}


def run_episode(task: str, retries: int = 2) -> float:
    for attempt in range(retries + 1):
        try:
            r   = requests.post(f"{BASE_URL}/reset",
                                json={"task_name": task}, timeout=30)
            obs = r.json().get("observation", r.json())
            for _ in range(12):
                if obs.get("done"):
                    break
                action = _call_llm(obs)   # ✅ FIXED: was _llm(obs)
                r      = requests.post(f"{BASE_URL}/step",
                                       json=action, timeout=30)
                obs    = r.json().get("observation", r.json())
            return float(obs.get("episode_score", 0.0))
        except Exception:
            if attempt == retries:
                return 0.0
            time.sleep(2 ** attempt)
    return 0.0


def main(output_json: bool = False):
    tasks   = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
    results = {}
    for task in tasks:
        scores = [run_episode(task) for _ in range(RUNS)]
        results[task] = {
            "mean_score": round(sum(scores) / len(scores), 4),
            "scores":     [round(s, 4) for s in scores],
            "runs":       RUNS,
        }
    if output_json:
        print(json.dumps(results))
    else:
        print("\n=== CredLess-Env Baseline ===")
        for t, r in results.items():
            bar = "█" * int(r["mean_score"] * 20)
            print(f"  {t:25s} [{bar:<20}] {r['mean_score']:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", action="store_true")
    main(p.parse_args().output_json)