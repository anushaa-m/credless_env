from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import argparse

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAVE_PATH = Path(__file__).parent / "model.pkl"
FEATURE_NAMES_PATH = Path(__file__).parent / "feature_names.txt"
METADATA_PATH = Path(__file__).parent / "metadata.json"
METRICS_PATH = Path(__file__).parent / "metrics.json"
RANDOM_SEED = 42

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pipeline import FEATURE_NAMES, clean_dataset, engineer_features, prepare_model_frame


def to_default_risk_target(target: pd.Series) -> pd.Series:
    # The raw dataset target behaves like a "good outcome" label where 1 is safer.
    # The oracle must predict default risk, so invert it here.
    return (1 - target.astype(int)).astype(int)


def load_split(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = clean_dataset(pd.read_csv(csv_path))
    features = engineer_features(frame)
    target = to_default_risk_target(frame["target"])
    return features, target


def curriculum_order(features_df: pd.DataFrame) -> pd.Index:
    risk_proxy = (
        0.24 * features_df["total_delinquency_score"]
        + 0.18 * features_df["debt_burden_score"]
        + 0.10 * features_df["overdraft_risk"]
        + 0.08 * features_df["location_risk_index"]
        + 0.16 * (1.0 - features_df["payment_reliability"])
        + 0.14 * (1.0 - features_df["income_capacity_score"])
        + 0.10 * (1.0 - features_df["employment_stability"])
    ).clip(0.0, 1.0)
    return np.abs(risk_proxy - 0.5).sort_values(ascending=False).index


def curriculum_stages(X_train_df: pd.DataFrame, y_train: pd.Series) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    ordered_idx = curriculum_order(X_train_df)
    X_ord = X_train_df.loc[ordered_idx]
    y_ord = y_train.loc[ordered_idx]
    n = len(X_ord)
    stage1 = max(1, int(n * 0.40))
    stage2 = max(stage1, int(n * 0.75))
    return [
        ("stage1_easy", X_ord.iloc[:stage1], y_ord.iloc[:stage1]),
        ("stage2_medium", X_ord.iloc[:stage2], y_ord.iloc[:stage2]),
        ("stage3_hard", X_ord, y_ord),
    ]


def best_threshold(model: LogisticRegression, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, float]:
    probs = model.predict_proba(X_val.values)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val.values, probs)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def metrics_for_split(model: LogisticRegression, X: pd.DataFrame, y: pd.Series, threshold: float) -> dict[str, float]:
    probs = model.predict_proba(X.values)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y.values, probs)),
        "accuracy": float(accuracy_score(y.values, preds)),
        "precision": float(precision_score(y.values, preds, zero_division=0)),
        "recall": float(recall_score(y.values, preds, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CredLess oracle logistic-regression baseline.")
    parser.add_argument("--train-csv", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=DATA_DIR / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=DATA_DIR / "test.csv")
    return parser.parse_args()


def train() -> dict[str, object]:
    args = parse_args()

    print("=" * 62)
    print("  CredLess oracle training")
    print("=" * 62)

    use_prepared_splits = args.train_csv.exists() and args.val_csv.exists()
    if use_prepared_splits:
        print("\n[1/5] Loading prepared train/val/test splits ...")
        X_tr_df, y_tr = load_split(args.train_csv)
        X_val_df, y_val = load_split(args.val_csv)
        has_test = args.test_csv.exists()
        if has_test:
            X_te_df, y_te = load_split(args.test_csv)
        else:
            X_te_df, y_te = X_val_df, y_val
        source = "prepared_splits"
    else:
        print("\n[1/5] Prepared splits missing. Falling back to full dataset split ...")
        X_all, y_all, _ = prepare_model_frame()
        y_all = to_default_risk_target(y_all)
        X_tr_df, temp_df, y_tr, temp_y = train_test_split(
            X_all,
            y_all,
            test_size=0.30,
            random_state=RANDOM_SEED,
            stratify=y_all,
        )
        X_val_df, X_te_df, y_val, y_te = train_test_split(
            temp_df,
            temp_y,
            test_size=0.50,
            random_state=RANDOM_SEED,
            stratify=temp_y,
        )
        has_test = True
        source = "fallback_split"

    print(f"  Train rows      : {len(X_tr_df):,}")
    print(f"  Validation rows : {len(X_val_df):,}")
    print(f"  Test rows       : {len(X_te_df):,}")
    print(f"  Features        : {len(FEATURE_NAMES)}")

    print("\n[2/5] Training logistic-regression curriculum baseline ...")
    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=4000,
        solver="lbfgs",
    )
    for stage_name, X_stage, y_stage in curriculum_stages(X_tr_df, y_tr):
        print(f"  {stage_name:<14} rows={len(X_stage):,}")
        model.fit(X_stage.values, y_stage.values)

    print("\n[3/5] Selecting validation threshold ...")
    threshold, val_best_f1 = best_threshold(model, X_val_df, y_val)
    val_metrics = metrics_for_split(model, X_val_df, y_val, threshold)
    print(f"  Threshold       : {threshold:.4f}")
    print(f"  Val AUC         : {val_metrics['auc']:.4f}")
    print(f"  Val Accuracy    : {val_metrics['accuracy']:.4f}")
    print(f"  Val Precision   : {val_metrics['precision']:.4f}")
    print(f"  Val Recall      : {val_metrics['recall']:.4f}")

    print("\n[4/5] Evaluating test split ...")
    test_metrics = metrics_for_split(model, X_te_df, y_te, threshold)
    print(f"  Test AUC        : {test_metrics['auc']:.4f}")
    print(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision  : {test_metrics['precision']:.4f}")
    print(f"  Test Recall     : {test_metrics['recall']:.4f}")
    test_probs = model.predict_proba(X_te_df.values)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    print("\n  Classification report (test):")
    print(classification_report(y_te.values, test_preds, target_names=["repay", "default"]))

    print("\n  Top positive coefficients:")
    coefs = model.coef_[0]
    ranked = sorted(zip(FEATURE_NAMES, coefs), key=lambda item: item[1], reverse=True)
    for name, value in ranked[:10]:
        print(f"    {name:<30} {value:>9.4f}")

    print("\n  Top negative coefficients:")
    for name, value in ranked[-10:]:
        print(f"    {name:<30} {value:>9.4f}")

    print("\n[5/5] Saving artifacts ...")
    low_risk_threshold = max(0.10, min(float(threshold) - 0.05, float(threshold) * 0.60))
    medium_risk_threshold = max(low_risk_threshold + 0.05, min(0.90, float(threshold)))
    metrics = {
        "validation_auc": round(val_metrics["auc"], 6),
        "validation_accuracy": round(val_metrics["accuracy"], 6),
        "validation_precision": round(val_metrics["precision"], 6),
        "validation_recall": round(val_metrics["recall"], 6),
        "validation_best_f1": round(val_best_f1, 6),
        "test_auc": round(test_metrics["auc"], 6),
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_precision": round(test_metrics["precision"], 6),
        "test_recall": round(test_metrics["recall"], 6),
    }
    artifact = {
        "model": model,
        "model_name": "LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "threshold": float(threshold),
        "metrics": metrics,
        "risk_thresholds": {
            "low_risk": round(low_risk_threshold, 4),
            "medium_risk": round(medium_risk_threshold, 4),
        },
        "train_rows": int(len(X_tr_df)),
        "validation_rows": int(len(X_val_df)),
        "test_rows": int(len(X_te_df)),
        "source": source,
        "positive_class_meaning": "default_risk",
        "raw_dataset_target_mapping": {"raw_target_1": "non_default_or_good_outcome", "model_target_1": "default_risk"},
    }
    metadata = {
        "model_name": "LogisticRegression",
        "feature_names": list(FEATURE_NAMES),
        "curriculum": ["stage1_easy", "stage2_medium", "stage3_hard"],
        "threshold": round(float(threshold), 6),
        "risk_thresholds": artifact["risk_thresholds"],
        "splits": {
            "train_rows": int(len(X_tr_df)),
            "validation_rows": int(len(X_val_df)),
            "test_rows": int(len(X_te_df)),
            "source": source,
        },
        "raw_dataset_target_mapping": artifact["raw_dataset_target_mapping"],
    }

    joblib.dump(artifact, SAVE_PATH)
    FEATURE_NAMES_PATH.write_text("\n".join(FEATURE_NAMES) + "\n", encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  Saved -> {SAVE_PATH}")
    print(f"  Saved -> {FEATURE_NAMES_PATH}")
    print(f"  Saved -> {METADATA_PATH}")
    print(f"  Saved -> {METRICS_PATH}")

    return artifact


if __name__ == "__main__":
    train()
