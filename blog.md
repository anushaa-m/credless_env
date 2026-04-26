# CredLess-Env: Training an AI Loan Officer for the 1.7 Billion People Banks Ignore

> **The invisible borrower problem:** 1.7 billion adults worldwide are unbanked. Not because they can't repay — but because traditional credit bureaus have no record of them. A student's UPI history, a gig worker's salary consistency, a first-time earner's overdraft behavior — none of it counts. CredLess-Env is a reinforcement learning environment that changes this. Instead of a CIBIL score, it trains an AI agent to act as a loan officer who reads behavioral financial signals and makes explainable lending decisions.

**Live:** [huggingface.co/spaces/anushaa-m/credless-env](https://huggingface.co/spaces/anushaa-m/credless-env) | **Code:** [github.com/anushaa-m/credless_env](https://github.com/anushaa-m/credless_env)

---

## Section 1: The Problem — Why Credit Scoring Is Broken

- Traditional FICO/CIBIL scoring requires loan history to produce a loan score — a circular trap for first-time borrowers.
- In India alone, ~190 million working adults have mobile banking and UPI activity but no formal credit file.
- These people are not high-risk. They are **unscored**. That's a different problem, and it requires a different tool.
- CredLess-Env replaces the credit bureau lookup with 20 behavioral features drawn from 42 raw signals — transaction patterns, salary consistency, overdraft behavior, digital payment usage, medical stress — all available from bank account data without requiring prior loan history.

---

## Section 2: What CredLess-Env Actually Is

CredLess-Env is a fully custom [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant RL environment. An AI agent plays the role of a loan officer. Each episode presents an applicant profile. The agent investigates, gathers evidence, and delivers a decision.

**Three tasks, increasing difficulty:**

| Task | Difficulty | What the agent sees | What it must do | Max steps |
|---|---|---|---|---|
| `binary_decision` | Easy | Full profile | Approve or deny | 8 |
| `risk_tiering` | Medium | Full profile | Assign tier + suggest credit limit | 8 |
| `adaptive_inquiry` | Hard | Partial profile (4–8 fields visible) | Request more fields, then decide | 8 |

- In `adaptive_inquiry`, the agent starts with only 4 fields visible. It can request more (first 2 are free; each additional costs −0.03 reward).
- The optimal strategy: reveal the 2–3 most decisive features, then decide. Agents that over-investigate burn reward on unnecessary requests.
- **Why 150,000 unique profiles?** Memorization is the silent killer of RL benchmarks. At 150K profiles (perfectly balanced, 50/50 default/non-default), no two episodes are identical. The agent must generalize, not memorize.

---

## Section 3: The Multi-Agent Architecture

CredLess-Env runs three specialized agents in a pipeline. No single model sees everything — each agent does one job and does it well.

### Agent 1 — The Oracle (Ground Truth)

The Oracle is a **Logistic Regression model** trained on 150,000 applicant profiles across 20 engineered features. It is the ground truth. Every agent decision is graded against it.

**What the Oracle learned (top feature coefficients by absolute magnitude):**

| Feature | Coefficient | What it means |
|---|---|---|
| `payment_reliability` | −11.25 | Strong payment history aggressively lowers default risk |
| `transaction_health` | −3.42 | Clean transaction patterns reduce risk |
| `location_risk_index` | −2.45 to −4.09 | Geographic credit environment matters |
| `employment_stability` | −1.76 | Job tenure reduces default probability |
| `account_maturity` | −1.16 | Older accounts are lower risk |

Negative coefficients mean the feature pushes toward **approval** (lower default probability). The Oracle doesn't just predict — it explains every decision via logistic coefficient contributions, giving the agent SHAP-style reasoning without SHAP overhead.

**Oracle metrics (test set, 22,500 holdout rows):**

- **Test AUC: 0.856** (validation: 0.866)
- **Test Accuracy: 75.8%** | **Precision: 71.4%** | **Recall: 85.9%**
- **Optimal decision threshold: 0.365** (F1-tuned on validation)
- **Risk tiers:** low\_risk < 0.219 → approve | 0.219–0.365 → medium\_risk → conditional\_approve | > 0.365 → deny

This threshold was learned, not hardcoded. The training script runs `precision_recall_curve` on the validation set and picks the threshold that maximizes F1.

**Curriculum training:** The Oracle wasn't trained on shuffled data. It learned in three stages:

1. `stage1_easy` (40% of data) — the most unambiguous cases, far from the decision boundary
2. `stage2_medium` (75%) — adds borderline cases
3. `stage3_hard` (100%) — all 105,000 training rows

Cases are ordered by `abs(risk_proxy − 0.5)` — the further from the decision boundary, the earlier they appear. This mirrors how human experts learn: start with obvious cases, graduate to hard ones.

### Agent 2 — The Decision Policy

The policy agent synthesizes the Oracle's risk score, the top feature contributions, and the market state to produce an APPROVE/DENY decision with reasoning.

- **Backbone:** `Qwen/Qwen2.5-0.5B-Instruct` (lightweight, runs on CPU during inference)
- **SFT baseline:** SGDClassifier trained on oracle-labeled samples
- **RL fine-tuning:** GRPO via TRL/Unsloth on `(prompt, response, reward)` trajectories
- **Online updates:** after every rollout, `update_policy(risk, decision, reward)` applies a gradient step — the policy is never static

**What a GRPO training sample looks like:**

```json
{
  "prompt": "payment_reliability=0.91, transaction_health=0.84, oracle_risk=0.18, market=Stable Credit. Decide.",
  "response": "APPROVE: payment_reliability strong at 0.91, well below risk threshold 0.365. Oracle default_prob=0.18.",
  "reward": 0.787
}
```

### Agent 3 — The Auditor

The Auditor runs after every terminal decision. It doesn't just check whether the decision was correct — it checks *how* the agent reasoned.

**Auditor scoring formula:**

```
episode_score = 0.65 × oracle_alignment + 0.35 × reasoning_score − bias_penalty
```

**Breaking down each component:**

- `oracle_alignment` = 1.0 if decision matches Oracle, else 0.15
- `reasoning_score` = scales from 0.20 (reasoning < 30 chars) up to 1.0 based on: evidence terms cited, market context mentioned, fraud signals acknowledged, and a +0.20 **recency bonus** if the agent mentions historical context ("old delinquency", "recovery", "account age")
- `bias_penalty` = −0.35 per protected term (religion, caste, gender, marital status, ethnicity, race) found in the reasoning text

**Example audit result on a high-scoring episode:**

```json
{
  "oracle_alignment": 1.0,
  "reasoning_score": 0.85,
  "recency_reasoning_bonus": 0.20,
  "bias_penalty": 0.0,
  "score": 0.8225
}
```

### The Market Agent — Dynamic Thresholds

This is the part that makes CredLess-Env genuinely novel. The same applicant, evaluated under two different macroeconomic conditions, gets two different decisions — and both are correct.

**The four market scenarios:**

| Market | Risk multiplier | Threshold delta | Effect |
|---|---|---|---|
| Stable Credit | 1.00 | 0.00 | Neutral |
| Economic Boom | 0.92 | +0.05 | Lenders tolerate more risk |
| High Inflation | 1.08 | −0.03 | Debt burden and missed payments rise |
| Recession | 1.18 | −0.06 | Conservative — tighter approvals |

**Concrete example:** An applicant with `debt_burden_score=0.65` and `payment_reliability=0.72`:
- In an **Economic Boom**: risk multiplier 0.92 × base_prob → effective threshold rises → **APPROVE** (medium\_risk)
- In a **Recession**: risk multiplier 1.18 × base_prob → effective threshold falls → **DENY** (high\_risk)

The applicant didn't change. The world did. That's the innovation.

---

## Section 4: Feature Engineering — 42 Columns, 20 Features, 5 Buckets

The dataset has 42 raw columns (one is a duplicate, dropped). Version 1 of the pipeline used only 21 and averaged them into 8 hand-crafted features, capping AUC at ~0.76. Version 2 uses all 42, engineers 20 features in five domain buckets, and reaches AUC 0.856 on test.

**The 5 buckets and their composite formulas:**

**1. Delinquency** (the strongest FICO signals — v1 ignored these entirely)
```python
total_delinquency_score = 0.20 × late_30_59 + 0.30 × late_60_89 + 0.50 × late_90plus
```
Heavier weight on worse buckets. A single 90-day late payment is worth 2.5× a 30-day one.

**2. Debt Burden**
```python
debt_burden_score = 0.35 × debt_ratio + 0.25 × emi_ratio + 0.20 × medical_debt + 0.10 × medical_condition + 0.10 × real_estate
```

**3. Income & Wealth** (log-scaled to handle extreme values)
```python
income_capacity_score = 0.45 × log(income) + 0.30 × log(revenue) + 0.15 × profit_margin + 0.10 × savings_months
```

**4. Payment Behaviour** (directly computable from UPI data — no bank history required)
```python
payment_reliability = 0.50 × (1 − failed_txn_ratio) + 0.30 × utility_payment_ratio + 0.20 × rent_regular
overdraft_risk      = overdraft_count / bank_account_age_months
```

**5. Stability**
```python
employment_stability = 0.60 × months_employed_scaled + 0.40 × employment_type_score
```

**Why this matters for the unbanked:** `payment_reliability` and `overdraft_risk` are computable from any UPI-linked bank account. Someone with zero credit history but 18 months of clean UPI transactions has a strong signal. V1 threw that signal away. V2 keeps it.

**Dataset statistics:**
- **150,000 rows**, 0 nulls after cleaning, 0 duplicates
- **Perfect class balance:** 50% default / 50% non-default (stratified across all splits)
- Train: 105,000 | Validation: 22,500 | Test: 22,500
- 42 raw columns → 20 engineered features → AUC improvement: 0.76 → **0.856**

---

## Section 5: Training Evidence — Real Numbers

### RL Baseline (500 Episodes, PPO)

| Metric | Value |
|---|---|
| Mean episode reward | **0.5629** |
| Std deviation | 0.1588 |
| Approve rate | 60.6% |

### Batch-Wise Learning Curve

| Batch | Episodes | Mean Reward | Max Reward | Phase |
|---|---|---|---|---|
| 0 | 0–31 | 0.5721 | 0.8074 | Exploration |
| 32 | 32–63 | 0.5733 | 0.8667 | Exploration |
| 64 | 64–95 | 0.5131 | 0.8425 | Dip (boundary cases) |
| 96 | 96–127 | **0.6034** | 0.8371 | First convergence peak |
| 128 | 128–159 | 0.4874 | 0.8274 | Mid-training instability |
| 160 | 160–191 | 0.5506 | 0.7024 | Stabilization |
| 192 | 192–223 | 0.5921 | 0.8227 | Recovery |
| 224 | 224–255 | 0.5044 | 0.8292 | Exploration restart |
| 256 | 256–287 | 0.5965 | 0.8835 | Refinement |
| 288 | 288–319 | 0.5644 | 0.7921 | Refinement |
| 320 | 320–351 | 0.5319 | 0.8415 | Refinement |
| 352 | 352–383 | 0.6016 | 0.7577 | Convergence |
| 384 | 384–415 | 0.5897 | 0.8205 | Convergence |
| 416 | 416–447 | **0.6041** | **0.9367** | Peak batch |
| 448 | 448–479 | 0.5442 | 0.6665 | Tightening |
| 480 | 480–499 | 0.5850 | 0.8144 | Final |

The dip at batch 64 (mean 0.5131) is expected — this is when the agent starts encountering boundary cases (oracle risk near 0.365) where any decision carries cost. The peak at batch 416 (max 0.9367) shows the agent has learned to confidently handle clear-cut cases while reasoning its way through ambiguous ones.

### Inference Run (200 Episodes, Qwen2.5-0.5B)

| Metric | Value |
|---|---|
| Mean reward | **0.5122** |
| Std deviation | 0.4421 |
| Mean oracle score | **0.6957** |
| Approve rate | 36.5% |
| Deny rate | 57.5% |
| Escalate rate | 6.0% |
| Episodes with positive reward | **93.5%** |
| Oracle agreement (reported) | **78.5%** |
| Best episode reward | 0.8971 |
| Worst episode reward (escalate) | −0.96 |

**What these numbers mean:**
- **78.5% oracle agreement** — the policy matches the Oracle's binary decision in 157 of 200 episodes.
- **93.5% positive reward** — only 13 episodes ended with negative reward (all were timeout escalations).
- The 6% escalation rate shows the agent sometimes runs out of steps before deciding. The GRPO fine-tuning target is to eliminate these by learning to decide faster.

### Real Episode Trace — High Reward APPROVE

```
oracle_risk:    0.177  (below low_risk threshold 0.219)
decision:       APPROVE
dominant:       payment_reliability = −11.25 (strongly supports approval)
reward:         0.787
oracle_score:   0.8224
explanation:    Decision=approve tier=low_risk with default_risk=0.1776
```

### Real Episode Trace — High Reward DENY

```
oracle_risk:    0.626  (above medium_risk threshold 0.365)
decision:       DENY
dominant:       payment_reliability = −11.25 (supports approval, overridden by high overall risk)
reward:         0.787
oracle_score:   0.627
explanation:    Decision=deny tier=high_risk with default_risk=0.6265
```

### GRPO Post-Training Targets

| Metric | Baseline | GRPO Target |
|---|---|---|
| Mean reward | 0.5629 | **0.70+** |
| Oracle agreement | 78.5% | **85%+** |
| Mean episode length | ~5.1 steps | **~4.2 steps** |

GRPO works by upweighting high-reward trajectories and discarding low-reward ones. The policy learns to skip unnecessary field requests and commit to decisions faster — shorter episodes with higher scores.

---

## Section 6: API — How to Talk to CredLess-Env

Six endpoints, all stateless from the caller's side:

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status":"healthy"}` |
| `/reset` | POST | Starts a new episode, returns first observation |
| `/step` | POST | Executes one action, returns next observation + reward |
| `/state` | GET | Current episode metadata |
| `/tasks` | GET | All 3 tasks with full action schemas |
| `/docs` | GET | Interactive OpenAPI documentation |

**Start an episode:**
```bash
curl -X POST https://anushaa-m-credless-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "adaptive_inquiry", "seed": 42}'
```

**Execute a field request:**
```bash
curl -X POST https://anushaa-m-credless-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "episode_id": "...", "action": {"action_type": "request_info", "params": {"field": "total_delinquency_score"}, "reasoning": "Delinquency history is the strongest predictor."}}'
```

**Session locking:** Each session is backed by a JSON file in `.runtime/sessions/`. A file-based mutex with a 5-second timeout prevents concurrent requests from corrupting the same episode state. Two agents can run in parallel — they never collide.

---

## Section 7: Reward Design — Anti-Hacking Safeguards

The reward function was designed to prevent every known RL exploitation pattern:

| Event | Reward |
|---|---|
| Correct terminal decision | +0.65–0.90 (via auditor formula) |
| Valid field request (free) | +0.05 |
| Excess field requests (>2) | −0.03 each |
| Market query | +0.05 |
| Correct fraud flag | +0.15 |
| False fraud alarm | −0.10 |
| Protected attribute in reasoning | −0.35 per term |
| Duplicate action | −0.10 |
| Episode timeout (>8 steps) | −0.96 |

**Anti-exploitation design:**
- **Timeout penalty (−0.96):** prevents infinite field-request loops. The agent can't stall indefinitely.
- **Duplicate penalty (−0.10):** requesting the same field twice is penalized immediately.
- **Deterministic Oracle:** the Oracle's decision for a given applicant is always the same. There's no stochastic reward to exploit.
- **150K unique profiles:** the agent cannot memorize. It must generalize.
- **Bias detector:** the Auditor scans reasoning text for religion, caste, gender, marital status. Any hit triggers a −0.35 deduction.

---

## Section 8: Hackathon Criteria Map

### Innovation (40%)
- Three-agent pipeline (Oracle + Policy + Auditor) with live interaction between agents
- Partially observable investigation — fields start hidden in `adaptive_inquiry`
- 4 dynamic market scenarios that shift approval thresholds in real time
- 42 behavioral features from Indian lending data (UPI transactions, overdrafts, salary stability)
- GRPO fine-tuning on 4-bit quantized Qwen2.5-0.5B via Unsloth/TRL
- Auditor bias detection scans reasoning text for 6 protected attribute categories
- `conditional_approve` action: a nuanced "approve with conditions" beyond binary lending decisions
- Deception detection: `deception_level` injection + `verify_income` action for income lie detection

### Storytelling (30%)
- The invisible borrower: 1.7 billion people excluded not by risk, but by absence of record
- Same applicant, different outcomes under Economic Boom vs. Recession — macro context as a lending input
- SHAP-style explanations: every decision comes with human-readable feature contributions
- Before/after demo: baseline agent (greedy, no investigation) vs. GRPO agent (strategic field requests)
- Behavioral signals that actually exist in UPI data for every unbanked Indian

### Showing Improvement (20%)
- AUC: 0.76 (v1, 8 hand-averaged features) → **0.856** (v2, 20 engineered features from 42 columns)
- RL baseline: mean reward **0.5629** over 500 episodes
- Inference: **78.5% oracle agreement**, **93.5% positive reward rate** over 200 episodes
- Batch 416 peak: mean 0.6041, max **0.9367**
- GRPO targets: 0.70+ mean reward, 85%+ oracle agreement

### Reward Pipeline (10%)
- Timeout, duplicate, and loop penalties eliminate degenerate strategies
- Deterministic Oracle prevents reward hacking through stochastic exploitation
- 150K unique profiles make memorization impossible
- Auditor formula decomposes score into oracle\_alignment + reasoning\_score + recency\_bonus − bias\_penalty
- Session mutex prevents episode contamination in concurrent runs

---

## Section 9: What's Next

CredLess-Env is production-ready today. Three extensions are already designed:

1. **Deception injection** — `deception_level` parameter makes applicants lie about income. The agent must cross-check `stated_income` against `transaction_health` and flag inconsistencies. Infrastructure is already in `server/environment.py`.

2. **Portfolio context** — add the agent's running approval/denial history to the observation. An agent approving too many borderline cases in one session should become more conservative. Connects to Theme #2 (portfolio-level risk management).

3. **`conditional_approve` expansion** — the action already exists. Next step: require the agent to specify both interest rate and maximum loan amount, with penalties for mis-pricing relative to the Oracle's risk tier.

---

*Built with FastAPI · Logistic Regression · Qwen2.5-0.5B · TRL/GRPO · OpenEnv · 150,000 real lending profiles*

**GitHub:** https://github.com/anushaa-m/credless_env | **HF Space:** https://huggingface.co/spaces/anushaa-m/credless-env
