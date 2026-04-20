"""
train.py - FinVerse-compatible training entrypoint

Usage:
    python train.py
    python train.py --csv path/to/data.csv
    python train.py --csv missing.csv      # synthetic fallback
    python train.py --n_rows 12000         # synthetic fallback row count

Outputs:
    data/dataset.jsonl
    models/saved/finverse_model.pkl
    models/saved/model_meta.json
    models/saved/scaler.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from pipeline.preprocessor import load_and_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinVerse training pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument(
        "--n_rows",
        type=int,
        default=12000,
        help="Synthetic row count when CSV is missing or omitted",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val", type=float, default=0.20, help="Validation fraction")
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/dataset.jsonl",
        help="Output JSONL path",
    )
    return parser.parse_args()


def prob_to_tier(prob: float) -> str:
    if prob >= 0.70:
        return "A"
    if prob >= 0.45:
        return "B"
    return "C"


def prob_to_decision(prob: float) -> str:
    return "approve" if prob >= 0.50 else "reject"


def _build_model(seed: int):
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )

    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  FinVerse - Data -> Decision Pipeline (Training)")
    print("=" * 60)

    df_model, scaler = load_and_preprocess(
        csv_path=args.csv,
        n_synthetic=args.n_rows,
        output_jsonl=args.jsonl,
        seed=args.seed,
    )

    feature_cols = [column for column in df_model.columns if column != "target"]
    X = df_model[feature_cols].values
    y = df_model["target"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val,
        random_state=args.seed,
        stratify=y,
    )

    print(
        f"[trainer] Train: {len(X_train)} | Val: {len(X_val)} | "
        f"Approve rate: {y.mean():.2%}"
    )

    model = _build_model(args.seed)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    val_acc = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_prob)

    print("\n[trainer] -- Validation Metrics -------------------------")
    print(f"  Accuracy : {val_acc:.4f}")
    print(f"  ROC-AUC  : {val_auc:.4f}")
    print(f"\n{classification_report(y_val, y_pred, target_names=['reject', 'approve'])}")

    os.makedirs("models/saved", exist_ok=True)

    with open("models/saved/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "model_type": "lightgbm" if HAS_LGBM else "logistic_regression",
        "val_accuracy": round(float(val_acc), 4),
        "val_auc": round(float(val_auc), 4),
        "seed": int(args.seed),
    }
    with open("models/saved/finverse_model.pkl", "wb") as handle:
        pickle.dump(artifact, handle)

    metadata = {
        "feature_cols": feature_cols,
        "model_type": artifact["model_type"],
        "val_accuracy": round(float(val_acc), 4),
        "val_auc": round(float(val_auc), 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "approve_rate_train": round(float(y_train.mean()), 4),
        "target_positive_class": "approve",
        "jsonl_path": args.jsonl,
        "model_path": "models/saved/finverse_model.pkl",
        "scaler_path": "models/saved/scaler.pkl",
    }
    with open("models/saved/model_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    val_df = pd.DataFrame(
        {
            "prob": y_prob,
            "decision": [prob_to_decision(prob) for prob in y_prob],
            "risk_tier": [prob_to_tier(prob) for prob in y_prob],
            "target": y_val,
        }
    )

    print("\n[trainer] Model saved    -> models/saved/finverse_model.pkl")
    print("[trainer] Metadata saved -> models/saved/model_meta.json")
    print("[trainer] Scaler saved   -> models/saved/scaler.pkl")
    print(f"[trainer] Validation rows: {len(val_df)}")


if __name__ == "__main__":
    main()
