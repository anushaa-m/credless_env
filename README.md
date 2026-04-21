# FinVerse End-to-End Pipeline

This repo now exposes the drafted FinVerse-style flow end to end:

- preprocess real CSV or synthetic fallback
- train or load a saved model
- run predictions
- run the deterministic oracle
- score model outputs against oracle outputs
- report target metrics including reject recall

## Current Structure

```text
credless_env/
|-- credless_model/
|-- data/
|   |-- cd_updated.csv
|   |-- dataset.jsonl
|   `-- synthetic_generator.py
|-- env/
|-- models/
|   |-- __init__.py
|   |-- trainer.py
|   `-- saved/
|-- pipeline/
|   |-- oracle.py
|   |-- preprocessor.py
|   |-- reasoning.py
|   `-- scorer.py
|-- server/
|-- inference.py
|-- train.py
`-- pyproject.toml
```

## What Is Implemented

- `pipeline/oracle.py` exists and normalizes oracle outputs to FinVerse decision/tier semantics.
- `models/trainer.py` exists and owns model build, training, validation, save, and load logic.
- Root `inference.py` is the FinVerse end-to-end pipeline runner.
- Root `train.py` is the standalone training entrypoint using the same shared trainer module.
- Missing CSV fallback to synthetic data is active in both training and inference paths.
- Auto-training from `inference.py` is active when saved artifacts are missing or `--retrain` is passed.
- Packaging metadata is configured in `pyproject.toml` with CLI entrypoints for train and inference.

## Commands

Train on the bundled real dataset:

```powershell
.\venv\Scripts\python.exe train.py --csv data\cd_updated.csv
```

Train on a real 5000-row sample:

```powershell
.\venv\Scripts\python.exe train.py --csv data\cd_updated.csv --n_rows 5000
```

Run full inference on the bundled real dataset:

```powershell
.\venv\Scripts\python.exe inference.py --csv data\cd_updated.csv --samples 20
```

Run full inference on a real 5000-row sample:

```powershell
.\venv\Scripts\python.exe inference.py --csv data\cd_updated.csv --n_rows 5000 --samples 20
```

Run the fallback synthetic pipeline:

```powershell
.\venv\Scripts\python.exe inference.py --n_rows 5000 --samples 20
```

Installed script entrypoints after package install:

```powershell
finverse-train --csv data\cd_updated.csv --n_rows 5000
finverse-infer --csv data\cd_updated.csv --n_rows 5000 --samples 20
```

## Notes

- For real CSVs, `--n_rows` now limits the sampled evaluation/training rows instead of being ignored.
- `target` semantics are normalized to `1 = approve`, `0 = reject`.
- `inference.py` now prints target accuracy, ROC-AUC, reject recall, and approve recall.
- The synthetic fallback remains useful for smoke tests, but its approval rate is not representative of the balanced real dataset.
