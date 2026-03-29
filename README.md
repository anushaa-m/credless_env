---
title: credless-env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
---

# CredLess-Env 🏦

**An OpenEnv RL environment for AI-powered alternative credit scoring.**

> An agent acts as a loan officer, evaluating applicants who have no traditional
> credit history — using only behavioural financial signals.  
> Powered by CredLess, an explainable ML model achieving AUC ≈ 0.94.

[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-credless--env-blue)](https://huggingface.co/spaces/anushaa-m/credless-env)
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