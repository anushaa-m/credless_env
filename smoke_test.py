import requests

BASE = "http://127.0.0.1:7860"


def safe_json(response, label=""):
    if response.status_code != 200:
        print(f"\nFAILED [{label}] status={response.status_code}")
        print(response.text[:500])
        raise SystemExit(1)
    return response.json()


def run_task(task_name: str, final_action: str) -> None:
    print("=" * 50)
    print(f"TEST: {task_name}")
    print("=" * 50)

    response = requests.post(f"{BASE}/reset", json={"task_name": task_name})
    data = safe_json(response, "reset")
    obs = data["observation"]
    print(f"Reset OK | task={obs['task_name']} | applicant={obs['applicant']['applicant_id']}")
    print(f"Visible={list(obs['applicant']['profile'].keys())[:6]} | missing={len(obs['applicant']['missing_fields'])}")

    response = requests.post(f"{BASE}/step", json=final_action)
    result = safe_json(response, "final")
    print(f"Final OK | done={result['done']} | reward={result['reward']}")
    print(f"Info: {result['info']}")
    print()


run_task(
    "binary_decision",
    "APPROVE",
)

run_task(
    "risk_tiering",
    "APPROVE",
)

run_task(
    "adaptive_inquiry",
    "REJECT",
)

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
