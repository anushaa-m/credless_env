"""
inference.py - FinVerse end-to-end inference and evaluation

Run:
    python inference.py
    python inference.py --csv path/to/data.csv
    python inference.py --samples 20
    python inference.py --csv path/to/data.csv --n_rows 5000

What it does:
    1. Loads model (trains from scratch if not found)
    2. Loads real CSV or synthetic fallback data
    3. Runs model predictions
    4. Runs oracle on the same rows
    5. Scores predictions vs oracle
    6. Prints metrics and sample outputs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from data.synthetic_generator import generate_synthetic_data
from models.trainer import (
    META_PATH,
    MODEL_PATH,
    SCALER_PATH,
    load_saved_model,
    prob_to_decision,
    prob_to_tier,
    train_from_frame,
)
from pipeline.oracle import oracle_decision
from pipeline.preprocessor import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    _clip_outliers,
    _encode_categoricals,
    _fill_missing,
)
from pipeline.scorer import batch_score


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

TIER_COLOR = {"A": GREEN, "B": YELLOW, "C": RED}
DEC_COLOR = {"approve": GREEN, "reject": RED}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinVerse inference pipeline")
    parser.add_argument("--csv", type=str, default=None, help="Path to real CSV")
    parser.add_argument(
        "--n_rows",
        type=int,
        default=None,
        help="Synthetic row count, or real CSV sample size when provided",
    )
    parser.add_argument("--samples", type=int, default=20, help="Sample rows to print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true", help="Force retrain model")
    parser.add_argument("--jsonl", type=str, default="data/dataset.jsonl")
    return parser.parse_args()


def _approve_target_from_raw(value: object) -> int:
    return 0 if int(value) == 1 else 1


def _approve_target_from_value(value: object) -> int:
    return int(value)


def _load_raw_frame(csv_path: str | None, n_rows: int, seed: int) -> tuple[pd.DataFrame, str]:
    if csv_path and Path(csv_path).exists():
        raw_df = pd.read_csv(csv_path, low_memory=False)
        raw_df.columns = [column.strip().lower().replace(" ", "_") for column in raw_df.columns]
        raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]
        if n_rows and n_rows > 0 and len(raw_df) > n_rows:
            raw_df = raw_df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
        source = str(Path(csv_path))
    else:
        if csv_path:
            print(f"[inference] CSV not found at '{csv_path}' - using synthetic fallback")
        else:
            print("[inference] No CSV provided - using synthetic fallback")
        raw_df = generate_synthetic_data(n_samples=n_rows or 12000, seed=seed, include_target=True)
        source = "synthetic"

    return raw_df.reset_index(drop=True), source


def _build_ml_frame(raw_df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.copy()
    df = _fill_missing(df)
    df = _clip_outliers(df)
    df = _encode_categoricals(df)

    ml_features = [column for column in NUMERIC_COLS if column in df.columns]
    for column in CATEGORICAL_COLS:
        enc_col = f"{column}_enc"
        if enc_col in df.columns:
            ml_features.append(enc_col)
    ml_features = list(dict.fromkeys(ml_features))

    df_model = df[ml_features].copy()
    if "target" in df.columns:
        target_converter = _approve_target_from_raw if source != "synthetic" else _approve_target_from_value
        df_model["target"] = df["target"].apply(target_converter).astype(int).values
    else:
        df_model["target"] = 0
    return df_model, ml_features


def _ensure_model(args: argparse.Namespace, df_model: pd.DataFrame, feature_cols: list[str]):
    if args.retrain or not MODEL_PATH.exists() or not SCALER_PATH.exists() or not META_PATH.exists():
        if args.retrain:
            print("[inference] --retrain flag set - retraining model")
        else:
            print("[inference] No saved model found - training now")
        artifacts = train_from_frame(
            df_model,
            feature_cols,
            seed=args.seed,
            val_size=0.20,
            jsonl_path=args.jsonl,
            save=True,
        )
        print(
            f"[inference] trained model | "
            f"val_acc={artifacts.val_accuracy:.4f} | val_auc={artifacts.val_auc:.4f}"
        )
        return artifacts.model, artifacts.scaler, artifacts.feature_cols
    print("[inference] Loading saved model")
    return load_saved_model()


def _print_banner() -> None:
    print("\n" + "=" * 65)
    print(f"  {BOLD}{CYAN}FinVerse - Inference & Evaluation{RESET}")
    print("  Applicant Profile -> Model -> Oracle -> Score")
    print("=" * 65)


def _print_sample(i: int, row_raw: dict, model_pred: dict, oracle_result: dict, score: float) -> None:
    model_decision = model_pred["decision"]
    model_tier = model_pred["risk_tier"]
    model_conf = model_pred["confidence"]
    oracle_decision_value = oracle_result["decision"]
    oracle_tier = oracle_result["risk_tier"]
    oracle_conf = oracle_result["confidence"]
    match = "OK" if model_decision == oracle_decision_value else "NO"
    match_color = GREEN if model_decision == oracle_decision_value else RED

    income = float(row_raw.get("monthlyincome", 0) or 0)
    late_90 = int(float(row_raw.get("numberoftimes90dayslate", 0) or 0))
    debt = float(row_raw.get("debtratio", 0) or 0)
    failed_txn = float(row_raw.get("failed_txn_ratio", 0) or 0)

    print(
        f"\n  {BOLD}Sample {i + 1:>3}{RESET}  "
        f"Income=INR {income:>8,.0f}  Late90={late_90}  DebtR={debt:.2f}  FailTxn={failed_txn:.2f}"
    )
    print(
        f"    Model  -> {DEC_COLOR[model_decision]}{model_decision:<7}{RESET}  "
        f"Tier={TIER_COLOR[model_tier]}{model_tier}{RESET}  conf={model_conf:.3f}"
    )
    print(
        f"    Oracle -> {DEC_COLOR[oracle_decision_value]}{oracle_decision_value:<7}{RESET}  "
        f"Tier={TIER_COLOR[oracle_tier]}{oracle_tier}{RESET}  conf={oracle_conf:.3f}"
    )
    print(f"    Score  -> {match_color}{score:.4f}  {match}{RESET}")


def main() -> None:
    args = parse_args()
    _print_banner()
    started = time.time()

    print(f"\n{BOLD}[1/4] Loading Data{RESET}")
    raw_df, source = _load_raw_frame(args.csv, args.n_rows, args.seed)
    df_model, ml_features = _build_ml_frame(raw_df, source)

    print(f"\n{BOLD}[2/4] Loading / Training Model{RESET}")
    model, scaler, feature_cols = _ensure_model(args, df_model, ml_features)

    for column in feature_cols:
        if column not in df_model.columns:
            df_model[column] = 0.0

    X = df_model[feature_cols].values
    X_scaled = scaler.transform(X)
    y_true = df_model["target"].values

    print(f"\n{BOLD}[3/4] Running Predictions{RESET}")
    y_prob = model.predict_proba(X_scaled)[:, 1]
    model_predictions = [
        {
            "decision": prob_to_decision(prob),
            "risk_tier": prob_to_tier(prob),
            "confidence": round(float(prob), 4),
        }
        for prob in y_prob
    ]

    print(f"\n{BOLD}[4/4] Oracle + Scoring{RESET}")
    oracle_results = [oracle_decision(raw_df.iloc[idx].to_dict()) for idx in range(len(raw_df))]
    results = batch_score(model_predictions, oracle_results)

    total = len(model_predictions)
    sample_count = min(args.samples, total)
    sample_indices = np.random.default_rng(args.seed).choice(total, sample_count, replace=False)

    print(f"\n{'-' * 65}")
    print(f"  {BOLD}Sample Predictions (showing {sample_count} rows){RESET}")
    print(f"{'-' * 65}")

    for rank, idx in enumerate(sample_indices):
        _print_sample(
            rank,
            raw_df.iloc[idx].to_dict(),
            model_predictions[idx],
            oracle_results[idx],
            results["scores"][idx],
        )

    elapsed = time.time() - started
    gt_acc = accuracy_score(y_true, (y_prob >= 0.50).astype(int))
    gt_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    target_report = classification_report(
        y_true,
        (y_prob >= 0.50).astype(int),
        target_names=["reject", "approve"],
        output_dict=True,
        zero_division=0,
    )

    model_tiers = [pred["risk_tier"] for pred in model_predictions]
    oracle_tiers = [oracle["risk_tier"] for oracle in oracle_results]

    def tier_dist(tiers: list[str]) -> dict[str, str]:
        n = len(tiers)
        return {tier: f"{tiers.count(tier) / n:.1%}" for tier in ["A", "B", "C"]}

    model_dist = tier_dist(model_tiers)
    oracle_dist = tier_dist(oracle_tiers)
    model_approve = sum(1 for pred in model_predictions if pred["decision"] == "approve") / total
    oracle_approve = sum(1 for oracle in oracle_results if oracle["decision"] == "approve") / total

    print(f"\n{'=' * 65}")
    print(f"  {BOLD}{CYAN}FINAL METRICS{RESET}")
    print(f"{'=' * 65}")
    print(f"  Total samples evaluated : {total}")
    print(f"  Data source             : {source}")
    print(f"  Elapsed time            : {elapsed:.1f}s")
    print()
    print(f"  {BOLD}vs Target{RESET}")
    print(f"    Accuracy              : {GREEN}{gt_acc:.4f}{RESET}")
    if not np.isnan(gt_auc):
        print(f"    ROC-AUC               : {GREEN}{gt_auc:.4f}{RESET}")
    print(f"    Reject recall         : {GREEN}{target_report['reject']['recall']:.4f}{RESET}")
    print(f"    Approve recall        : {GREEN}{target_report['approve']['recall']:.4f}{RESET}")
    print()
    print(f"  {BOLD}vs Oracle{RESET}")
    print(f"    Decision accuracy     : {GREEN}{results['decision_acc']:.4f}{RESET}")
    print(f"    Tier accuracy         : {GREEN}{results['tier_acc']:.4f}{RESET}")
    print(f"    Avg composite score   : {GREEN}{results['avg_score']:.4f}{RESET}")
    print(f"    Min / Max score       : {min(results['scores']):.4f} / {max(results['scores']):.4f}")
    print()
    print(f"  {BOLD}Tier Distribution{RESET}")
    print(f"    Model  -> A:{model_dist['A']}  B:{model_dist['B']}  C:{model_dist['C']}")
    print(f"    Oracle -> A:{oracle_dist['A']}  B:{oracle_dist['B']}  C:{oracle_dist['C']}")
    print()
    print(f"  {BOLD}Approve Rate{RESET}")
    print(f"    Model  : {model_approve:.2%}")
    print(f"    Oracle : {oracle_approve:.2%}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
