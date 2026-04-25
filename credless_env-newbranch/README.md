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

Train Agent 2 with **real GRPO (TRL + Unsloth)**:

```powershell
.\venv\Scripts\python.exe -m rl.trainer --algorithm grpo --episodes 256 --batch-size 32 --require-trl
```

Artifacts:

- `agent2_llm_checkpoints/grpo/`: fine-tuned LLM policy (loaded automatically by `pipeline/main_pipeline.py` if present)
- `rl/training_summary.json`: mean/std reward, approve-rate, decision distribution
- `rl/reward_curve.csv` (and `rl/reward_curve.png` if matplotlib is installed)
- `rl/reward_log.jsonl`: per-episode/step diagnostics

## Adaptive inquiry (hackathon demo)

The OpenEnv-style environment (`server/environment.py`) is **multi-step and partially observable**:

- The agent can `request_info`, `query_market`, or `flag_fraud`
- Only then does it commit to a terminal `approve` / `deny` (or `escalate`)
- Terminal reward is intentionally simple and demo-friendly:
  - **+ oracle_match**
  - **+ confidence bonus (via oracle confidence weighting)**
  - **- efficiency penalty** (for extra information gathering)

## Notes

- `credless_model/model.pkl` remains single source for Agent 1 inference.
- `inference.py` evaluates environment episodes, not separate supervised artifact files.
- Root CLI no longer writes `models/saved/*`.
