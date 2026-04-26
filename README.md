# CredLess Two-Agent Stack

This repo exposes one unified CredLess stack:

- train CredLess Agent 1 oracle model in `credless_model/`
- run two-agent inference with Agent 1 + Agent 2
- serve OpenEnv-compatible multi-step investigation environment
- score terminal decisions with shared oracle and grader logic

At the environment level, each episode is a 5-8 step lending investigation over
20 engineered real-world features with market context, applicant/action
history, and adversarial behavior such as fabricated or withheld fields.

## Current Structure

```text
credless_env/
|-- credless_model/
|-- data/
|-- env/
|-- models/
|   `-- __init__.py
|-- pipeline/
|   |-- main_pipeline.py
|   |-- oracle.py
|   `-- reasoning.py
|-- server/
|-- inference.py
|-- train.py
`-- pyproject.toml
```

## What Is Implemented

- `credless_model/train.py` owns Agent 1 training and artifact generation.
- `pipeline/main_pipeline.py` loads only CredLess Agent 1 artifact from `credless_model/model.pkl`.
- Root `train.py` delegates directly to CredLess trainer.
- Root `inference.py` runs two-agent CredLess evaluation against `CreditAnalystEnvironment`.
- `server/` owns live OpenEnv environment, oracle, and grading path.

## Dynamic Risk-Based Oracle

CredLess uses a macro-prudential risk layer instead of a fixed decision boundary.
The ground-truth oracle now mutates the default-risk threshold using the hidden
market state:

```text
dynamic_default_threshold = base_threshold / market_risk_index
```

This keeps the decision semantics correct for default probability: a higher
market risk index lowers the acceptable default-risk cutoff and makes approvals
more conservative. A lower index relaxes the cutoff during favorable markets.

| Market State | Risk Index | Effective Threshold | Analyst Stance |
| --- | ---: | ---: | --- |
| Economic Boom | 0.92 | ~0.40 | Aggressive: higher approval tolerance |
| Stable Credit | 1.00 | ~0.37 | Neutral baseline |
| Recession | 1.18 | ~0.31 | Conservative: lower default-risk tolerance |

The agent should query market conditions before a terminal approval or denial.
The same applicant can be approved in a boom and denied in a recession because
the oracle threshold changes with the hidden macro state.

`inference.py` logs `market_state`, `market_risk_index`, `base_threshold`, and
`dynamic_threshold` for every episode so before/after market-aware behavior can
be compared directly.

## Reward-Hacking Protection: Threshold Persistence

The standalone training entrypoint now guarantees `risk_thresholds` are saved in
`credless_model/model.pkl` (`low_risk`, `medium_risk`) using trained values. This
prevents server-oracle fallback defaults (`0.40`, `0.70`) and keeps tiered
decisions aligned with training-time calibration, improving observable inference
metrics such as reject recall and ROC-AUC.

## New Terminal Action: `conditional_approve`

CredLess now supports nuanced terminal decisions:

- `approve` for clear low-risk approvals
- `conditional_approve` for medium-risk ("yellow-zone") applicants with terms
- `deny` for high-risk applicants

`conditional_approve` accepts:

```json
{
  "action_type": "conditional_approve",
  "params": { "rate": 14.5, "max_amount": 50000 },
  "reasoning": "Approved with elevated rate given overdraft_risk"
}
```

Reward shaping is anti-hacking by design: medium-risk conditional approvals score
better than a wrong hard decision, while misuse on clear low/high-risk profiles
is penalized.

## Auditor Agent Oversight

CredLess now runs a dedicated `AuditorAgent` that wraps deterministic
`audit_terminal_action()` logic and performs explicit bias checks on reasoning.
Each terminal decision is reviewed by the auditor, and both `auditor_score` and
`audit_history` are returned in the observation so Agent 2 can learn to be right
for the right reasons (not just maximize raw reward).

## Applicant Lie Detection (2-Sigma Check)

CredLess now injects explicit income-lie adversarial cases and requires evidence
verification before terminal decisions.

- High-difficulty episodes can include inflated `stated_income`.
- The environment compares `stated_income` against transaction-health-inferred
  income and computes a discrepancy z-score.
- Discrepancy `> 2 sigma` is treated as a lie candidate.
- Agent can call `flag_fraud` (income rationale) or `verify_income` to register
  detection.

Reward wiring:

- `+0.5` for correctly detecting a `>2 sigma` lie before final decision
- `-0.3` for false positive income-lie flags
- `-1.0` missed-lie penalty if agent approves without detecting a true `>2 sigma` lie

## Delinquency time-decay (recency vs FICO-style static counts)

FinVerse down-weights **stale** delinquency and overdraft signals using
`bank_account_age_months` as a temporal proxy: recent problems keep more of
their impact; old events on a mature account are discounted before the oracle
runs `predict_proba`. Each applicant carries a `credit_trajectory` summary
(also requestable via `request_info`: `last_delinquency_months_ago`,
`bank_account_age_months`, `overdraft_count`). **Pitch:** unlike a static
bureau snapshot, recency weighting rewards recovery and long clean runs.

The `AuditorAgent` awards a small reasoning bonus when the rationale explicitly
references recency, stability, or historical vs recent behavior.

## Commands

Train CredLess Agent 1:

```powershell
.\venv\Scripts\python.exe train.py
```

Run local two-agent inference:

```powershell
.\venv\Scripts\python.exe inference.py --n-rows 20
```

Installed script entrypoints after package install:

```powershell
credless-train
credless-infer --n-rows 20
```

## Notes

- `credless_model/model.pkl` remains single source for Agent 1 inference.
- `inference.py` evaluates environment episodes, not separate supervised artifact files.
- Root CLI no longer writes `models/saved/*`.
