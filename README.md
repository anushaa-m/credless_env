---
title: CredLess-Env
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - finance
  - credit-scoring
license: mit
---



# CredLess-Env 🏦

**OpenEnv RL environment — AI Credit Analyst for alternative credit scoring.**

An agent acts as a loan officer evaluating applicants who have no traditional
credit history, using only behavioural financial signals. Powered by CredLess,
an explainable ML model (Logistic Regression, AUC ≈ 0.93).

[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-credless--env-blue)](https://huggingface.co/spaces/your-username/credless-env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Motivation

Traditional credit scoring excludes millions of people — students, gig workers,
first-time earners — because they lack loan history. CredLess-Env trains agents
to evaluate **behavioural financial signals** instead, making credit decisions
more inclusive and explainable.

---

## Environment Description

The agent plays a loan officer. Each episode presents an applicant profile.
The agent must assess the profile and decide: approve, deny, or assign a risk tier.
In the hardest task, only partial information is shown and the agent must
strategically request more data before deciding.

The environment uses a trained Logistic Regression model as a ground-truth oracle
to score every agent decision.

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

| Field | Type | Used in |
|---|---|---|
| `action_type` | string | all actions |
| `decision` | string | binary_decision, adaptive_inquiry |
| `tier` | string | risk_tiering |
| `credit_limit` | float (INR) | risk_tiering |
| `field_name` | string | adaptive_inquiry (request_field) |

---

## Observation Space
```json
{
  "applicant_id":      "A3F9C1",
  "revealed_fields":   {"transaction_activity": 0.82, "payment_consistency": 0.91, "...": "..."},
  "hidden_fields":     ["digital_usage", "salary_consistency"],
  "task_name":         "adaptive_inquiry",
  "step_reward":       0.05,
  "cumulative_reward": 0.10,
  "done":              false,
  "message":           "Revealed: 'account_age' = 48.2000",
  "episode_score":     0.0
}
```

| Field | Description |
|---|---|
| `applicant_id` | Unique ID for this episode's applicant |
| `revealed_fields` | Dict of visible feature name → float value |
| `hidden_fields` | List of fields available to request (adaptive_inquiry only) |
| `task_name` | Active task |
| `step_reward` | Reward for this specific step |
| `cumulative_reward` | Total reward so far this episode |
| `done` | True when episode has ended |
| `message` | Human-readable feedback from the environment |
| `episode_score` | Final grader score 0.0–1.0 (populated when done=True) |

### Feature Definitions

| Feature | Range | Description |
|---|---|---|
| `transaction_activity` | 0–1 | Normalised transaction volume score |
| `payment_consistency` | 0–1 | On-time payment ratio |
| `account_stability` | 0–1 | Balance stability score |
| `overdraft_count` | 0–20 | Overdraft events in last 12 months |
| `digital_usage` | 0–1 | UPI / digital payment adoption ratio |
| `salary_consistency` | 0–1 | Regularity of salary credits |
| `failed_tx_ratio` | 0–0.5 | Failed transactions / total |
| `account_age` | 1–120 months | Account age |

---

## Tasks

| Task | Difficulty | Description | Grader |
|---|---|---|---|
| `binary_decision` | 🟢 Easy | Full profile shown. Approve or deny. | Binary 1.0 / 0.0 |
| `risk_tiering` | 🟡 Medium | Assign tier + suggest credit limit in INR. | Tier accuracy 60% + limit error 40% |
| `adaptive_inquiry` | 🔴 Hard | Partial profile. Request fields (3 free, −0.10 each after), then decide. | Correctness 70% + efficiency 30% |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns `{"status":"healthy"}` |
| `/reset` | POST | Start new episode. Body: `{"task_name":"binary_decision"}` |
| `/step` | POST | Execute action. Body: CreditAction JSON |
| `/state` | GET | Current episode metadata |
| `/tasks` | GET | List all 3 tasks with action schemas |
| `/grader` | GET | Current episode grader status |
| `/baseline` | GET | Run baseline inference, return scores JSON |
| `/docs` | GET | Interactive OpenAPI documentation |

---

## Setup and Usage

### Install
```bash
pip install git+https://huggingface.co/spaces/your-username/credless-env
```

### Run locally
```bash
# 1. Clone and install
git clone https://huggingface.co/spaces/your-username/credless-env
cd credless-env
pip install -r requirements.txt

# 2. Train the model
python credless_model/train.py

# 3. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# 4. Test it
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

### Run with Docker
```bash
docker build -t credless-env:latest -f server/Dockerfile .
docker run -d -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e HF_TOKEN=hf_... \
  credless-env:latest
```

### Run inference
```bash
# Set env vars
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=hf_...
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

### Validate OpenEnv spec
```bash
openenv validate
```

---

## Baseline Scores

Scores below produced by `inference.py` using `meta-llama/Llama-3.1-8B-Instruct`
via HF router, 5 runs per task. **Update these with your real run results.**

| Task | Difficulty | Mean Score | Runs |
|---|---|---|---|
| `binary_decision` | Easy | — | 5 |
| `risk_tiering` | Medium | — | 5 |
| `adaptive_inquiry` | Hard | — | 5 |

> Run `python inference.py` with valid env vars to generate real scores.

---

## Reward Function

| Event | Reward |
|---|---|
| Valid field request (adaptive_inquiry) | +0.05 |
| Invalid / unknown field name | −0.05 |
| Duplicate field request | −0.10 |
| Wrong task context for request | −0.05 |
| Correct final decision | +1.0 (or partial for risk_tiering) |
| Wrong final decision | 0.0 |
| Episode timeout (≥12 steps) | −0.5 |

---

## Project Structure
```
credless-env/
├── inference.py           ← Main hackathon inference script
├── baseline.py            ← Legacy baseline (kept for compatibility)
├── models.py              ← Typed Action / Observation / State (Pydantic)
├── client.py              ← OpenEnv WebSocket client
├── openenv.yaml           ← OpenEnv metadata manifest
├── pyproject.toml
├── requirements.txt
├── README.md
├── .dockerignore
├── credless_model/
│   ├── train.py           ← Trains + saves model.pkl
│   └── model.pkl          ← Saved Logistic Regression pipeline
└── server/
    ├── app.py             ← FastAPI server
    ├── environment.py     ← RL episode logic
    ├── oracle.py          ← CredLess model wrapper
    ├── data_generator.py  ← Synthetic applicant generator
    ├── graders.py         ← Task graders (deterministic, 0.0–1.0)
    ├── tasks.py           ← Task registry
    └── Dockerfile
```