# CredLess-Env / FinVerse Compatibility Layer

This repository is still primarily an OpenEnv-style credit analyst environment, but it now also includes a FinVerse-compatible data pipeline layer for preprocessing, oracle logic, reasoning, scoring, training, and synthetic fallback data generation.

The important distinction is:

- `server/` and the root `inference.py` power the environment-style workflow.
- `pipeline/`, `data/synthetic_generator.py`, and root `train.py` are the FinVerse-compatible compatibility layer.

## Current Structure

```text
credless_env/
├── data/
│   ├── cd_updated.csv
│   ├── dataset.jsonl
│   └── synthetic_generator.py
├── pipeline/
│   ├── preprocessor.py
│   ├── reasoning.py
│   ├── scorer.py
│   └── __init__.py
├── server/
│   ├── app.py
│   ├── environment.py
│   ├── graders.py
│   ├── oracle.py
│   └── data_generator.py
├── credless_model/
│   ├── dataset_pipeline.py
│   ├── train.py
│   └── model.pkl
├── train.py
├── inference.py
├── models.py
├── requirements.txt
└── README.md
```

## What Exists Now

### FinVerse-compatible pieces

- `pipeline/preprocessor.py`
  - loads CSV data
  - falls back to synthetic generation if no CSV is provided or the CSV is missing
  - can write `data/dataset.jsonl`
  - returns `(df_model, scaler)` for model training
- `pipeline/oracle.py`
  - deterministic oracle interface with:
    - `decision`: `approve` / `reject`
    - `risk_tier`: `A` / `B` / `C`
    - `confidence`
- `pipeline/reasoning.py`
  - deterministic feature-grounded reasoning strings
- `pipeline/scorer.py`
  - weighted evaluation with normalization for:
    - `approve/reject`
    - `approve/deny`
    - `A/B/C`
    - `low_risk/medium_risk/high_risk`
- `data/synthetic_generator.py`
  - generates synthetic feature rows
  - unlabeled by default
  - only adds oracle-derived labels when `include_target=True`
- `train.py`
  - trains a compatibility model
  - writes:
    - `data/dataset.jsonl`
    - `models/saved/finverse_model.pkl`
    - `models/saved/model_meta.json`
    - `models/saved/scaler.pkl`

### Existing environment pieces

- `server/oracle.py`
  - contains the environment oracle class
  - also contains a deterministic raw-row helper `oracle_decision(...)`
- `inference.py`
  - currently runs the environment / agent loop
  - it is not the standalone FinVerse end-to-end evaluation script from the draft pipeline layout

## What Does Not Exist Yet

These are still not fully implemented in the original draft shape:

- `models/trainer.py`
- a standalone FinVerse-style `inference.py` that does:
  - preprocess
  - train/load model
  - run predictions
  - run oracle
  - score model vs oracle

Those behaviors were partially implemented through compatibility files instead of a full package refactor.

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py
python train.py --csv path/to/data.csv
python train.py --csv path/to/data.csv --n_rows 12000
python train.py --n_rows 500
```

Outputs:

- `data/dataset.jsonl`
- `models/saved/finverse_model.pkl`
- `models/saved/model_meta.json`
- `models/saved/scaler.pkl`

Behavior:

- CSV provided and found: train on real CSV
- CSV provided and missing: synthetic fallback
- no CSV provided: synthetic fallback
- `--n_rows` controls synthetic fallback size

Training labels are normalized to FinVerse approval semantics:
  - `1 = approve`
  - `0 = reject`

## Preprocessing

Example:

```python
from pipeline.preprocessor import load_and_preprocess

df_model, scaler = load_and_preprocess(
    csv_path=None,
    output_jsonl="data/dataset.jsonl",
)
```

Behavior:

- `csv_path` exists: load that CSV
- `csv_path` missing: warn and use synthetic fallback
- no `csv_path`: use synthetic fallback

If you want the bundled real dataset explicitly, pass `--csv data/cd_updated.csv`.

## Synthetic Data

Example:

```python
from data.synthetic_generator import generate_synthetic_data

df = generate_synthetic_data(n_samples=1000)
df_labeled = generate_synthetic_data(n_samples=1000, include_target=True)
```

Safety note:

- unlabeled is the default to reduce oracle-label leakage
- `include_target=True` is explicit because oracle-labeled synthetic data can make offline evaluation look artificially strong if misused

## Scoring

Example:

```python
from pipeline.scorer import evaluate_prediction

score = evaluate_prediction(
    pred={"decision": "approve", "risk_tier": "B", "confidence": 0.61},
    oracle={"decision": "approve", "risk_tier": "A", "confidence": 0.71},
)
```

Weights:

- decision match: `0.60`
- tier match / distance: `0.25`
- confidence alignment: `0.15`

## Environment

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

Run the current root inference client:

```bash
python inference.py
```

This `inference.py` is for the environment workflow, not the standalone FinVerse pipeline draft.

## Important Accuracy Notes

- The repo is now hybrid.
- The compatibility layer is real and runnable.
- The original draft README was not fully accurate for this codebase.

Specifically:

- there is no `models/trainer.py`
- root `inference.py` is not the draft pipeline runner
- the repo remains hybrid: environment-first, with a working FinVerse compatibility layer added on top

If you want the repo to exactly match the original FinVerse folder contract, the next step is a deliberate refactor rather than more README changes.
