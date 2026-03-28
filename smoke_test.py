# smoke_test.py
import requests

BASE = "http://127.0.0.1:8000"

def safe_json(r, label=""):
    if r.status_code != 200:
        print(f"\n❌ FAILED [{label}] status={r.status_code}")
        print(r.text[:500])
        exit(1)
    return r.json()

print("=" * 50)
print("TEST 1: binary_decision")
print("=" * 50)

r   = requests.post(f"{BASE}/reset", json={"task_name": "binary_decision"})
d   = safe_json(r, "reset")
obs = d["observation"]
print(f"✅ Reset OK  | task={obs['task_name']} | applicant={obs['applicant_id']}")
print(f"   Revealed fields: {list(obs['revealed_fields'].keys())}")

# ✅ FIX: send action fields at top level, NOT wrapped in {"action": ...}
r   = requests.post(f"{BASE}/step", json={"action_type": "approve", "decision": "approve"})
d   = safe_json(r, "step")
obs = d["observation"]
print(f"✅ Step OK   | done={obs['done']} | score={obs['episode_score']}")
print(f"   Message: {obs['message']}")

print()
print("=" * 50)
print("TEST 2: risk_tiering")
print("=" * 50)

r   = requests.post(f"{BASE}/reset", json={"task_name": "risk_tiering"})
d   = safe_json(r, "reset")
obs = d["observation"]
print(f"✅ Reset OK  | task={obs['task_name']} | applicant={obs['applicant_id']}")

r   = requests.post(f"{BASE}/step", json={
    "action_type": "assign_tier",
    "tier": "low_risk",
    "credit_limit": 75000.0
})
d   = safe_json(r, "step")
obs = d["observation"]
print(f"✅ Step OK   | done={obs['done']} | score={obs['episode_score']}")
print(f"   Message: {obs['message']}")

print()
print("=" * 50)
print("TEST 3: adaptive_inquiry")
print("=" * 50)

r   = requests.post(f"{BASE}/reset", json={"task_name": "adaptive_inquiry"})
d   = safe_json(r, "reset")
obs = d["observation"]
print(f"✅ Reset OK  | task={obs['task_name']}")
print(f"   Visible : {list(obs['revealed_fields'].keys())}")
print(f"   Hidden  : {obs['hidden_fields']}")

r   = requests.post(f"{BASE}/step", json={
    "action_type": "request_field",
    "field_name": "account_age"
})
d   = safe_json(r, "request_field")
obs = d["observation"]
print(f"✅ Request OK | reward={obs['step_reward']} | {obs['message']}")

r   = requests.post(f"{BASE}/step", json={
    "action_type": "deny",
    "decision": "deny"
})
d   = safe_json(r, "final step")
obs = d["observation"]
print(f"✅ Final OK  | done={obs['done']} | score={obs['episode_score']}")
print(f"   Message: {obs['message']}")

print()
print("=" * 50)
print("✅ ALL TESTS PASSED")
print("=" * 50)