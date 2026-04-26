# CredLess 🧠💳

**Behavioral RL for Fair Credit Decision-Making**

> *An open reinforcement learning environment that learns trust from behavior — not just documents.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Environment](https://img.shields.io/badge/Env-OpenEnv-orange)](https://anushaa-m-credless-env.hf.space)
[![Blog](https://img.shields.io/badge/Blog-Hugging%20Face-yellow)](https://huggingface.co)

---

## What Is CredLess?

CredLess is an **open RL training and evaluation environment** for credit decision-making that works with alternative behavioral signals instead of — or in addition to — traditional credit scores.

It trains an agent to behave like a thoughtful credit analyst: gathering evidence step-by-step, weighing behavioral signals under uncertainty, detecting fraud, and making decisions that are audited for bias and reasoning quality.

The system is designed around one core question:

> *Can we measure financial trust differently — and more fairly?*

---

## Hackathon Theme Alignment

### 🎯 AI for Financial Inclusion
Traditional credit infrastructure excludes hundreds of millions of people globally — not because they're untrustworthy, but because they're **undocumented**. CredLess directly addresses this by:
- Operating on **alternative behavioral signals** (payment reliability, income capacity, employment stability, overdraft patterns) that don't require formal credit history
- Rewarding agents that make correct decisions for **underbanked applicants** using non-standard evidence
- Building explainability into every decision via SHAP + an auditor agent

### 🤖 Reinforcement Learning for Real-World Decision-Making
CredLess is not a classification model. It's a **sequential decision process** modeled as an RL environment where:
- The agent operates under **partial observability** (not all applicant fields are visible at step 0)
- Actions have consequences: each information request costs a step penalty
- The reward function combines accuracy, reasoning quality, fraud detection, and efficiency
- Policy optimization runs via **GRPO** (Unsloth + TRL) with a lightweight **PPO** fallback

### 🛡️ Responsible AI / Fairness
The environment has **bias detection baked into the reward signal**, not bolted on afterward:
- An `AuditorAgent` reviews every terminal decision
- Any reasoning that mentions protected attributes (caste, religion, gender, ethnicity, marital status, race) reduces the auditor score by 0.5 — a hard penalty
- Fraud flags are validated against ground truth to prevent false accusations

---

## Environment Design

### Architecture

```
CredLess-Env (OpenEnv Server)
│
├── /reset  →  Initializes episode with partial applicant profile
├── /step   →  Accepts action JSON, returns observation + reward + done
│
├── CreditAnalystEnvironment
│   ├── Oracle          (ground truth default probability + decision)
│   ├── AuditorAgent    (reasoning quality + bias detection)
│   ├── RiskPredictor   (SHAP-powered live risk score)
│   └── DataGenerator   (synthetic applicant profiles with adversarial variants)
│
└── Tasks
    ├── binary_decision
    ├── risk_tiering
    └── adaptive_inquiry
```

### Action Space

| Action | Description | Reward Effect |
|---|---|---|
| `request_info` | Reveal a hidden applicant field | +0.05 − step cost |
| `query_market` | Reveal current market conditions | +0.05 − step cost |
| `flag_fraud` | Raise a fraud flag with a reason | +0.05–0.50 or −0.10–0.30 depending on accuracy |
| `verify_income` | Trigger income discrepancy check | Routes to `flag_fraud` with income context |
| `approve` | Terminal: approve the application | Oracle-aligned reward |
| `conditional_approve` | Terminal: approve with rate + amount cap | Reward adjusted for term quality |
| `deny` | Terminal: reject the application | Oracle-aligned reward |
| `escalate` | Terminal: escalate for human review | Fixed partial reward (−0.5) |

### Observation Space

Each step returns a `FinVerseObservation` containing:
- `applicant` — visible profile fields, missing fields, declared income, credit trajectory
- `market_state` — base lending rate, default risk index, sector outlook (if queried)
- `current_policy` — required fields and max DTI for this episode
- `fraud_flags_raised` — list of flags submitted so far
- `portfolio_context` — session-level approve rate and avg risk (anti-gaming signal)
- `compliance_history` — recent auditor scores
- `step`, `max_steps`, `cumulative_reward` — episode progress signals

### Reward Function

```
episode_reward = 
    0.65 × task_score             # oracle alignment + tier match
  + 0.35 × auditor_score          # reasoning quality + bias check
  − efficiency_penalty            # excess field requests + skipped market query
  + fraud_bonus                   # correct fraud detection on adversarial applicants
  − false_alarm_penalty           # fraud flags on clean applicants
  − missed_income_lie_penalty     # income discrepancy not caught before approval
  − repeat_action_penalty         # anti-hacking: duplicate actions penalized
  − timeout_penalty               # no terminal decision within MAX_STEPS=8
```

---

## Three Tasks

### `binary_decision` (Easy)
The agent must approve or deny. Required fields: `payment_reliability`, `debt_burden_score`. Baseline task for establishing decision accuracy.

### `risk_tiering` (Medium)
The agent must classify the applicant into a risk tier (low / medium / high) and optionally offer a `conditional_approve` with an interest rate and loan cap. Required fields expand to include `overdraft_risk`.

### `adaptive_inquiry` (Hard)
Full complexity. The agent faces adversarial applicants who fabricate income or withhold delinquency history. Required fields include `total_delinquency_score`. Income verification via 2-sigma discrepancy detection is available. The agent must detect lies before approving — or pay a 1.0 penalty for missing them.

---

## Behavioral Signals Used

These are the alternative features CredLess operates on — none require a traditional credit file:

| Signal | Meaning |
|---|---|
| `payment_reliability` | On-time payment consistency |
| `income_capacity_score` | Earning capacity derived from transaction patterns |
| `employment_stability` | Job tenure and sector consistency |
| `overdraft_risk` | Frequency and magnitude of overdrafts |
| `total_delinquency_score` | Composite delinquency metric |
| `debt_burden_score` | Total obligations vs. capacity |
| `account_maturity` | Depth and age of banking relationship |
| `medical_stress_score` | Health-related financial shocks |
| `stated_income` + `transaction_health` | Cross-verified income (fraud signal) |

---

## Training Setup

### GRPO (Primary, via Unsloth + TRL)

Used when GPU + TRL backend is available. The LLM (default: `Qwen/Qwen2.5-0.5B-Instruct`) is fine-tuned in 4-bit precision. A custom `reward_fn` scores completions by matching agent decisions against oracle-preferred actions weighted by trajectory reward.

```python
from rl.trainer import RLTrainer, RLTrainingConfig

config = RLTrainingConfig(
    algorithm="grpo",
    episodes=256,
    batch_size=32,
    base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    learning_rate=1e-5,
)
trainer = RLTrainer(config=config)
summary = trainer.train(users)
```

### PPO (Lightweight Fallback)

A pure NumPy PPO implementation with GAE, clipped surrogate objective, entropy regularization, and value function training. Runs on CPU with no dependencies beyond NumPy. Weights saved as `.npz`.

```bash
# Train
python env/env_runner.py --mode ppo-train --episodes 50 --task adaptive_inquiry

# Evaluate
python env/env_runner.py --mode ppo-eval --episodes 20 --weights saved/ppo_policy.npz
```

### Dynamic Approve Rate Control

If the batch approve rate exceeds 65%, a dynamic penalty scales up proportionally and is subtracted from rewards of approved decisions in that batch. This prevents the agent from gaming the reward by rubber-stamping all applications.

```python
if approve_rate > 0.65:
    penalty = (approve_rate - 0.65) * 0.5
    for t in trajectories:
        if t.summary["decision"] == "APPROVE":
            t.total_reward -= penalty
```

---

## Metrics to Track

During training, log and plot:

| Metric | What It Shows |
|---|---|
| Mean episode reward | Overall decision quality |
| Approve rate per batch | Policy balance — not over-approving |
| Oracle alignment | Accuracy vs. ground truth |
| Auditor score | Reasoning and bias quality |
| Policy entropy | Exploration vs. exploitation tradeoff |
| Policy loss | PPO/GRPO optimization signal |

<!-- INSERT: Reward curve across episodes -->
<!-- INSERT: Approve rate stability plot -->
<!-- INSERT: Entropy decay curve -->

---

## Quickstart

```bash
git clone https://github.com/your-username/credless
cd credless
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add HF_TOKEN and ENV_BASE_URL

# Run a single rollout
python env/env_runner.py --mode rollout --task binary_decision

# RL training loop
python rl/trainer.py --algorithm grpo --episodes 256 --batch-size 32

# Lightweight PPO (no GPU needed)
python env/env_runner.py --mode ppo-train --episodes 30
```

**Live environment:** `https://anushaa-m-credless-env.hf.space`

---

## Production Roadmap

### Phase 1 — Data Integration
- UPI / PhonePe / BHIM transaction streams → `payment_reliability` proxy
- GST filing consistency → `income_capacity_score`
- Telco and utility payment data for completely unbanked users
- DigiLocker document signals for identity anchoring

### Phase 2 — Deployment
- FastAPI wrapper around CredLess-Env for production inference
- Target: < 200ms per decision step (4-bit quantized LLM via Unsloth)
- JSON API: decision + risk score + SHAP explanation per request
- Docker image with health checks and horizontal scaling support

### Phase 3 — Trust & Monitoring
- SHAP explanations surfaced to loan officers in plain language
- Full audit logs per episode: reward components, oracle alignment, bias flags
- Human escalation path via the built-in `escalate` action
- Scheduled bias audits across demographic cohorts
- Drift detection on approve rate and signal distribution

### Phase 4 — Scale
- NBFC / fintech API integration as a drop-in alternative scoring layer
- Regional model variants (state-level income norms, sector risk indices)
- Federated fine-tuning support for institution-specific lending policy
- Regulator-ready audit trail export (RBI / SEBI compliance format)

---

## Repository Structure

```
credless/
├── env/
│   └── env_runner.py          # PPO trainer + rollout runner + LLM policy
├── rl/
│   ├── trainer.py             # GRPO/PPO training loop
│   ├── rollout_collector.py   # Trajectory collection + policy update
│   └── reward_logger.py       # JSONL reward logging
├── server/
│   ├── environment.py         # CreditAnalystEnvironment (OpenEnv)
│   ├── graders.py             # AuditorAgent + reward evaluation
│   ├── oracle.py              # Ground truth oracle
│   ├── tasks.py               # Task definitions + difficulty
│   └── data_generator.py      # Synthetic applicant generation
├── pipeline/
│   └── main_pipeline.py       # CreditDecisionPipeline + FrozenRiskPredictor
├── data/
│   └── synthetic_generator.py
├── models.py                  # FinVerseAction / Observation / State schemas
├── credless.md                # Hugging Face blog post
└── README.md
```

---

## Why This Matters

Over 190 million adults in India alone are unbanked or underbanked. Most of them aren't credit risks — they're credit **invisibles**. They have behavioral track records of reliability, consistency, and trustworthiness that traditional systems have no vocabulary for.

CredLess doesn't lower the bar. It moves the bar to where it should have been.

A system that can see payment consistency, income trajectory, and employment stability — without requiring a CIBIL score — can reach the people who need credit access most: the home kitchen entrepreneur, the seasonal worker, the first-generation saver.

> **Fairness isn't about equal rules. It's about being seen — not just scored.**

---

## License

MIT — use it, extend it, improve it.

---

*Contributions welcome. If you're working on financial inclusion, alternative data, or RL for high-stakes decisions — let's talk.*
