"""
CredLess — train.py
===================
Trains the current oracle model on the engineered feature frame produced by
dataset_pipeline.py, now with a minimal easy→medium→hard curriculum stage.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [info] XGBoost not installed — using sklearn HistGradientBoostingClassifier instead.")
    print("         Install with:  pip install xgboost\n")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pipeline import prepare_model_frame

SAVE_PATH = Path(__file__).parent / "model.pkl"
FEATURE_NAMES_PATH = Path(__file__).parent / "feature_names.txt"
METADATA_PATH = Path(__file__).parent / "metadata.json"
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 3


def best_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def subgroup_audit(model, X_test, y_test, threshold, feature_names):
    ages = X_test[:, feature_names.index("age_years")]
    qs = np.percentile(ages, [25, 50, 75])
    labels = ["Q1 (youngest 25%)", "Q2", "Q3", "Q4 (oldest 25%)"]
    print("\n—— Subgroup parity audit (age quartiles) —————————————")
    print(f"  {'Group':<22} {'n':>5}  {'def%':>6}  {'AUC':>7}  {'F1':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}")
    for i, lbl in enumerate(labels):
        lo = qs[i - 1] if i > 0 else -np.inf
        hi = qs[i] if i < 3 else np.inf
        mask = (ages > lo) & (ages <= hi)
        if mask.sum() < 10:
            continue
        Xg, yg = X_test[mask], y_test[mask]
        probs = model.predict_proba(Xg)[:, 1]
        preds = (probs >= threshold).astype(int)
        auc = roc_auc_score(yg, probs) if len(np.unique(yg)) > 1 else float("nan")
        f1 = f1_score(yg, preds, zero_division=0)
        print(f"  {lbl:<22} {mask.sum():>5}  {yg.mean():>5.1%}  {auc:>7.4f}  {f1:>7.4f}")
    print()


def curriculum_order(features_df):
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


def curriculum_stages(X_train_df, y_train):
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


def build_candidates():
    if HAS_XGB:
        return {
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                use_label_encoder=False,
                random_state=RANDOM_SEED,
                n_jobs=1,
                tree_method="hist",
            )
        }
    return {
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        )
    }


def train():
    print("=" * 62)
    print("  CredLess model training")
    print("=" * 62)

    print("\n[1/6] Loading & engineering features ...")
    features, target, _ = prepare_model_frame()
    X_df, y = features.copy(), target.copy()
    fn = list(X_df.columns)
    print(f"  Rows            : {len(X_df):,}")
    print(f"  Features        : {len(fn)}  {fn}")
    print(f"  Default rate    : {y.mean():.1%}  (class 1={y.sum():,}, class 0={(y==0).sum():,})")

    print("\n[2/6] Splitting data (80/20 stratified) ...")
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_df,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train: {len(X_tr_df):,}  |  Test: {len(X_te_df):,}")

    print(f"\n[3/6] Benchmarking candidate models ({CV_FOLDS}-fold CV on train set) ...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    for name, model in build_candidates().items():
        scores = cross_validate(
            model,
            X_tr_df.values,
            y_tr.values,
            cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1"},
            n_jobs=1,
        )
        auc_scores = scores["test_auc"]
        f1_scores = scores["test_f1"]
        results[name] = {"auc": auc_scores.mean(), "f1": f1_scores.mean(), "auc_std": auc_scores.std()}
        print(f"  {name:<24}  AUC={auc_scores.mean():.4f} ±{auc_scores.std():.4f}  F1={f1_scores.mean():.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  Best model → {best_name}  (AUC {results[best_name]['auc']:.4f})")

    print(f"\n[4/6] Training {best_name} on full training split with curriculum ...")
    best_model = build_candidates()[best_name]
    for stage_name, X_stage, y_stage in curriculum_stages(X_tr_df, y_tr):
        print(f"  {stage_name:<14} rows={len(X_stage):,}")
        best_model.fit(X_stage.values, y_stage.values)

    threshold, best_f1_tr = best_threshold(best_model, X_tr_df.values, y_tr.values)
    print(f"  Optimal threshold : {threshold:.3f}  (train F1: {best_f1_tr:.4f})")

    print(f"\n[5/6] Held-out test evaluation ...")
    test_probs = best_model.predict_proba(X_te_df.values)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    test_auc = roc_auc_score(y_te, test_probs)

    print(f"\n  Test AUC-ROC : {test_auc:.4f}  (CV was {results[best_name]['auc']:.4f})")
    print(f"\n  Classification report (threshold={threshold:.3f}):")
    print(classification_report(y_te, test_preds, target_names=["repay", "default"]))

    tn, fp, fn_count, tp = confusion_matrix(y_te, test_preds).ravel()
    print("  Confusion matrix:")
    print(f"    True negatives  (repay → repay)    : {tn:>6}")
    print(f"    False positives (repay → default)  : {fp:>6}")
    print(f"    False negatives (default → repay)  : {fn_count:>6}")
    print(f"    True positives  (default → default): {tp:>6}")

    print("\n  Feature importances (top 20):")
    importances = best_model.feature_importances_
    ranked = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    for feat_name, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"    {feat_name:<30} {imp:.4f}  {bar}")

    subgroup_audit(best_model, X_te_df.values, y_te.values, threshold, fn)

    print("[6/6] Saving model artifact ...")
    low_risk_threshold = max(0.10, min(float(threshold) - 0.05, float(threshold) * 0.60))
    medium_risk_threshold = max(low_risk_threshold + 0.05, min(0.90, float(threshold)))
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "threshold": float(threshold),
        "feature_names": fn,
        "metrics": {
            "cv_auc": float(results[best_name]["auc"]),
            "test_auc": float(test_auc),
        },
        "risk_thresholds": {
            "low_risk": round(low_risk_threshold, 4),
            "medium_risk": round(medium_risk_threshold, 4),
        },
        "cv_auc": float(results[best_name]["auc"]),
        "test_auc": float(test_auc),
        "default_rate": float(y.mean()),
        "n_train": int(len(X_tr_df)),
        "n_test": int(len(X_te_df)),
    }
    metadata = {
        "model_name": best_name,
        "feature_names": fn,
        "curriculum": ["stage1_easy", "stage2_medium", "stage3_hard"],
        "threshold": round(float(threshold), 6),
        "risk_thresholds": artifact["risk_thresholds"],
        "metrics": artifact["metrics"],
        "default_rate": round(float(y.mean()), 6),
        "n_train": int(len(X_tr_df)),
        "n_test": int(len(X_te_df)),
    }

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, SAVE_PATH)
    FEATURE_NAMES_PATH.write_text("\n".join(fn) + "\n", encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved → {SAVE_PATH}")

    print("\n" + "=" * 62)
    print(f"  Final model  : {best_name}")
    print(f"  Test AUC     : {test_auc:.4f}")
    print(f"  Threshold    : {threshold:.3f}")
    print(f"  Features     : {len(fn)}")
    print("=" * 62)
    return artifact


if __name__ == "__main__":
    train()
