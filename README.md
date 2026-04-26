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
