import requests

BASE = "http://127.0.0.1:7860"


def safe_json(response, label=""):
    if response.status_code != 200:
        print(f"\nFAILED [{label}] status={response.status_code}")
        print(response.text[:500])
        raise SystemExit(1)
    return response.json()


def run_task(task_name: str, final_action: dict) -> None:
    print("=" * 50)
    print(f"TEST: {task_name}")
    print("=" * 50)

    response = requests.post(f"{BASE}/reset", json={"task_name": task_name})
    data = safe_json(response, "reset")
    obs = data["observation"]
    print(f"Reset OK | task={obs['task_name']} | applicant={obs['applicant']['applicant_id']}")
    print(f"Visible={list(obs['applicant']['profile'].keys())[:6]} | missing={len(obs['applicant']['missing_fields'])}")

    missing = obs["applicant"]["missing_fields"]
    if missing:
        response = requests.post(
            f"{BASE}/step",
            json={"action_type": "request_info", "params": {"field": missing[0]}, "reasoning": ""},
        )
        obs = safe_json(response, "request")["observation"]
        print(f"Request OK | reward={obs['step_reward']} | message={obs['message']}")

    response = requests.post(
        f"{BASE}/step",
        json={
            "action_type": "flag_fraud",
            "params": {"reason": "Confidence anomaly noted during review."},
            "reasoning": "",
        },
    )
    obs = safe_json(response, "fraud")["observation"]
    print(f"Fraud OK | flags={obs['fraud_flags_raised']} | reward={obs['step_reward']}")

    response = requests.post(f"{BASE}/step", json={"action_type": "query_market", "params": {}, "reasoning": ""})
    obs = safe_json(response, "market")["observation"]
    print(f"Market OK | visible={obs['market_visible']} | context={obs['market_state']}")

    response = requests.post(f"{BASE}/step", json=final_action)
    obs = safe_json(response, "final")["observation"]
    print(f"Final OK | done={obs['done']} | score={obs['episode_score']}")
    print(f"Message: {obs['message']}")
    print()


run_task(
    "binary_decision",
    {
        "action_type": "approve",
        "params": {"tier": "medium_risk", "rate": 12.0},
        "reasoning": "Payment reliability and debt burden remain acceptable after market review.",
    },
)

run_task(
    "risk_tiering",
    {
        "action_type": "approve",
        "params": {"tier": "medium_risk", "rate": 14.0},
        "reasoning": "Moderate risk profile with enough stability to approve on mid-tier terms.",
    },
)

run_task(
    "adaptive_inquiry",
    {
        "action_type": "deny",
        "params": {"tier": "high_risk", "rate": 18.5},
        "reasoning": "Observed uncertainty and adverse signals remain too high after review.",
    },
)

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
