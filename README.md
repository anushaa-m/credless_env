# CredLess-Env 🏦

**An OpenEnv RL environment for AI-powered alternative credit scoring.**

> An agent acts as a loan officer, evaluating applicants who have no traditional
> credit history — using only behavioural financial signals.  
> Powered by CredLess, an explainable ML model achieving AUC ≈ 0.94.

[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-credless--env-blue)](https://huggingface.co/spaces/your-username/credless-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Why This Environment?

Traditional credit scoring fails millions of people — students, gig workers,
first-time earners — because they lack borrowing history. CredLess-Env trains
agents to evaluate **behavioural financial signals** instead:

| Feature | Description |
|---|---|
| `transaction_activity` | Normalised transaction volume score (0–1) |
| `payment_consistency` | On-time payment ratio (0–1) |
| `account_stability` | Balance stability score (0–1) |
| `overdraft_count` | Overdraft events in last 12 months (0–20) |
| `digital_usage` | UPI/digital payment adoption ratio (0–1) |
| `salary_consistency` | Regularity of salary credits (0–1) |
| `failed_tx_ratio` | Failed transactions / total (0–0.5) |
| `account_age` | Account age in months (1–120) |

---

## Action Space
```json
{
  "action_type": "approve | deny | assign_tier | request_field",
  "decision":    "approve | deny",
  "tier":        "low_risk | medium_risk | high_risk",
  "credit_limit": 75000.0,
  "field_name":  "overdraft_count"
}
```

## Observation Space
```json
{
  "applicant_id":      "A3F9C1",
  "revealed_fields":   {"transaction_activity": 0.82, "payment_consistency": 0.91},
  "hidden_fields":     ["digital_usage", "salary_consistency"],
  "task_name":         "adaptive_inquiry",
  "step_reward":       0.05,
  "cumulative_reward": 0.10,
  "done":              false,
  "message":           "'overdraft_count' = 2.0000",
  "episode_score":     0.0
}
```

---

## Tasks

| Task | Difficulty | Description | Grader |
|---|---|---|---|
| `binary_decision` | 🟢 Easy | Full profile given. Approve or deny. | Binary 0.0 / 1.0 |
| `risk_tiering` | 🟡 Medium | Assign tier + suggest credit limit (INR). | Tier accuracy 60% + limit error 40% |
| `adaptive_inquiry` | 🔴 Hard | Partial profile. Request fields then decide. | Correctness 70% + efficiency 30% |

---

## Setup & Usage

### Install
```bash
pip install git+https://huggingface.co/spaces/your-username/credless-env
```

### Run locally
```bash
# 1. Train and save the model
python credless_model/train.py

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run with Docker
```bash
docker build -t credless-env .
docker run -d -p 7860:7860 -e OPENAI_API_KEY=sk-... credless-env
```

### Use the client (async)
```python
import asyncio
from client import CreditEnv
from models import CreditAction

async def main():
    async with CreditEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_name="binary_decision")
        print(result.observation.revealed_fields)
        result = await env.step(CreditAction(
            action_type="approve", decision="approve"
        ))
        print(f"Score: {result.observation.episode_score}")

asyncio.run(main())
```

### Use the client (sync)
```python
from client import CreditEnv
from models import CreditAction

with CreditEnv(base_url="http://localhost:7860").sync() as env:
    result = env.reset(task_name="risk_tiering")
    result = env.step(CreditAction(
        action_type="assign_tier",
        tier="low_risk",
        credit_limit=150000.0
    ))
    print(f"Score: {result.observation.episode_score}")
```

### Run baseline
```bash
export OPENAI_API_KEY=sk-...
python baseline.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/ws` | WebSocket | Persistent session (used by client) |
| `/health` | GET | Health check → `{"status": "healthy"}` |
| `/reset` | POST | Start new episode |
| `/step` | POST | Execute action |
| `/state` | GET | Current episode state |
| `/tasks` | GET | List all 3 tasks with action schemas |
| `/grader` | GET | Current episode grader status |
| `/baseline` | GET | Trigger baseline script → scores JSON |
| `/docs` | GET | Interactive OpenAPI docs |

---

## Baseline Scores

| Task | Model | Mean Score | Runs |
|---|---|---|---|
| `binary_decision` | gpt-4o-mini | 0.820 | 5 |
| `risk_tiering` | gpt-4o-mini | 0.641 | 5 |
| `adaptive_inquiry` | gpt-4o-mini | 0.573 | 5 |

---

## Project Structure
```
credless-env/
├── models.py                  ← Typed Action / Observation (Pydantic)
├── client.py                  ← OpenEnv WebSocket client
├── baseline.py                ← OpenAI API inference script
├── openenv.yaml               ← OpenEnv metadata manifest
├── pyproject.toml             ← Package definition
├── README.md
├── credless_model/
│   ├── train.py               ← Trains + saves model.pkl
│   └── model.pkl              ← Saved Logistic Regression pipeline
└── server/
    ├── app.py                 ← FastAPI server + extra endpoints
    ├── environment.py         ← RL episode logic
    ├── oracle.py              ← CredLess model wrapper
    ├── data_generator.py      ← Synthetic applicant generator
    ├── graders.py             ← Task graders (deterministic, 0.0–1.0)
    ├── tasks.py               ← Task registry
    ├── requirements.txt
    └── Dockerfile
```