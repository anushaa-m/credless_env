# baseline.py  — run with: python baseline.py
# Requires: OPENAI_API_KEY env var
#           ENV_BASE_URL  env var (default http://localhost:7860)
import os
import json
import argparse
import requests
from openai import OpenAI
import time
from dotenv import load_dotenv
load_dotenv()   # ✅ loads .env file for local dev

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MODEL    = os.getenv("BASELINE_MODEL", "gpt-4o-mini")
RUNS     = int(os.getenv("BASELINE_RUNS", "5"))   # episodes per task

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """You are a credit analyst AI. Evaluate applicants and
return ONLY a valid JSON action object — no extra text."""

TASK_HINTS = {
    "binary_decision": (
        'Return {"action_type":"approve","decision":"approve"} '
        'or {"action_type":"deny","decision":"deny"}'
    ),
    "risk_tiering": (
        'Return {"action_type":"assign_tier",'
        '"tier":"low_risk|medium_risk|high_risk","credit_limit":<INR float>}'
    ),
    "adaptive_inquiry": (
        'You may request hidden fields first: '
        '{"action_type":"request_field","field_name":"<name>"}. '
        'Then decide: {"action_type":"approve","decision":"approve|deny"}'
    ),
}


def _call_llm(obs: dict) -> dict:
    task  = obs.get("task_name", "binary_decision")
    hint  = TASK_HINTS.get(task, "")
    user  = (
        f"Task: {task}\n"
        f"Applicant profile: {json.dumps(obs.get('revealed_fields', {}), indent=2)}\n"
        f"Hidden fields available: {obs.get('hidden_fields', [])}\n"
        f"Environment message: {obs.get('message', '')}\n\n"
        f"Hint: {hint}\n"
        "Reply with ONLY one JSON action object."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def run_episode(task: str, retries: int = 2) -> float:
    for attempt in range(retries + 1):
        try:
            r   = requests.post(f"{BASE_URL}/reset",
                                json={"task_name": task}, timeout=30)
            obs = r.json().get("observation", r.json())
            for _ in range(12):
                if obs.get("done"):
                    break
                action = _llm(obs)
                r      = requests.post(f"{BASE_URL}/step",
                                       json=action, timeout=30)
                obs    = r.json().get("observation", r.json())
            return float(obs.get("episode_score", 0.0))
        except Exception as e:
            if attempt == retries:
                return 0.0
            time.sleep(2)  # wait before retry
    return 0.0

def main(output_json: bool = False):
    tasks   = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
    results = {}

    for task in tasks:
        scores = []
        for _ in range(RUNS):
            try:
                scores.append(run_episode(task))
            except Exception as e:
                scores.append(0.0)
                if not output_json:
                    print(f"  [WARN] {task} episode failed: {e}")

        results[task] = {
            "mean_score": round(sum(scores) / len(scores), 4),
            "scores":     [round(s, 4) for s in scores],
            "runs":       RUNS,
        }

    if output_json:
        print(json.dumps(results))          # consumed by /baseline endpoint
    else:
        print("\n=== CredLess-Env Baseline Results ===")
        for task, r in results.items():
            bar = "█" * int(r["mean_score"] * 20)
            print(f"{task:25s} [{bar:<20}] {r['mean_score']:.3f}  {r['scores']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-json", action="store_true",
                   help="Print JSON to stdout (used by /baseline endpoint)")
    args = p.parse_args()
    main(args.output_json)