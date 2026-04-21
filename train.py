"""
train.py - FinVerse-compatible training entrypoint

Usage:
    python train.py
    python train.py --csv path/to/data.csv
    python train.py --csv missing.csv      # synthetic fallback
    python train.py --n_rows 12000         # synthetic fallback row count
    python train.py --csv path/to/data.csv --n_rows 5000

Outputs:
    data/dataset.jsonl
    models/saved/finverse_model.pkl
    models/saved/model_meta.json
    models/saved/scaler.pkl
"""

from __future__ import annotations

import argparse

import pandas as pd

from models.trainer import prob_to_decision, prob_to_tier, train_from_frame
from pipeline.preprocessor import load_and_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinVerse training pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument(
        "--n_rows",
        type=int,
        default=None,
        help="Synthetic row count, or real CSV sample size when provided",
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


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  FinVerse - Data -> Decision Pipeline (Training)")
    print("=" * 60)

    df_model, _ = load_and_preprocess(
        csv_path=args.csv,
        n_synthetic=args.n_rows or 12000,
        max_rows=args.n_rows if args.csv else None,
        output_jsonl=args.jsonl,
        seed=args.seed,
    )

    feature_cols = [column for column in df_model.columns if column != "target"]
    X = df_model[feature_cols].values
    y = df_model["target"].values

    split_idx = int(round(len(X) * (1.0 - args.val)))
    train_count = max(split_idx, 0)
    val_count = max(len(X) - train_count, 0)

    print(
        f"[trainer] Train: {train_count} | Val: {val_count} | "
        f"Approve rate: {y.mean():.2%}"
    )

    artifacts = train_from_frame(
        df_model,
        feature_cols,
        seed=args.seed,
        val_size=args.val,
        jsonl_path=args.jsonl,
        save=True,
    )

    print("\n[trainer] -- Validation Metrics -------------------------")
    print(f"  Accuracy : {artifacts.val_accuracy:.4f}")
    print(f"  ROC-AUC  : {artifacts.val_auc:.4f}")
    print(f"\n{artifacts.classification_report_text}")

    val_df = pd.DataFrame(
        {
            "prob": artifacts.y_prob,
            "decision": [prob_to_decision(prob) for prob in artifacts.y_prob],
            "risk_tier": [prob_to_tier(prob) for prob in artifacts.y_prob],
            "target": artifacts.y_val,
        }
    )

    print("\n[trainer] Model saved    -> models/saved/finverse_model.pkl")
    print("[trainer] Metadata saved -> models/saved/model_meta.json")
    print("[trainer] Scaler saved   -> models/saved/scaler.pkl")
    print(f"[trainer] Validation rows: {len(val_df)}")


if __name__ == "__main__":
    main()
