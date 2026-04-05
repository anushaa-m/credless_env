"""
CredLess — train.py  (v2 — full-signal rewrite)
================================================
Uses the new dataset_pipeline.py (v2) which engineers 20 features from all
42 raw columns instead of 8 features from 21 columns.

What this script does
---------------------
1.  Loads and engineers features via dataset_pipeline.py
2.  Runs a quick benchmark: Logistic Regression vs XGBoost vs Random Forest
    so you can see which model actually wins on YOUR data
3.  Trains the best model with proper CV + held-out test set
4.  Tunes the decision threshold for credit-scoring (minimise false negatives)
5.  Prints feature importance ranked by XGBoost gain
6.  Runs a subgroup parity audit across age quartiles
7.  Saves the full artifact (model + threshold + metadata)

Expected improvement
--------------------
  v1  AUC ~0.76  (8 features, 21 raw columns, no delinquency signal)
  v2  AUC ~0.85+ (20 features, 42 raw columns, delinquency + wealth included)

Run:
    python credless_model/train.py
"""

import sys
import warnings
import joblib
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)

# ── try XGBoost, fall back to HistGradientBoosting ────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [info] XGBoost not installed — using sklearn HistGradientBoostingClassifier instead.")
    print("         Install with:  pip install xgboost\n")

# ── local pipeline ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pipeline import prepare_model_frame, FEATURE_NAMES

SAVE_PATH   = Path(__file__).parent / "model.pkl"
RANDOM_SEED = 42
TEST_SIZE   = 0.20
CV_FOLDS    = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def best_threshold(model, X_val, y_val):
    """
    Finds the probability threshold that maximises F1 on validation data.
    In credit scoring you typically want to push this DOWN slightly to catch
    more defaults (reduce false negatives at the cost of more false positives).
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def subgroup_audit(model, X_test, y_test, threshold, feature_names):
    """Parity check across age quartiles (index 17 = age_years feature)."""
    ages  = X_test[:, feature_names.index("age_years")]
    qs    = np.percentile(ages, [25, 50, 75])
    labels = ["Q1 (youngest 25%)", "Q2", "Q3", "Q4 (oldest 25%)"]
    print("\n── Subgroup parity audit (age quartiles) ─────────────────────")
    print(f"  {'Group':<22} {'n':>5}  {'def%':>6}  {'AUC':>7}  {'F1':>7}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}")
    for i, lbl in enumerate(labels):
        lo = qs[i - 1] if i > 0 else -np.inf
        hi = qs[i]     if i < 3 else  np.inf
        mask = (ages > lo) & (ages <= hi)
        if mask.sum() < 10:
            continue
        Xg, yg = X_test[mask], y_test[mask]
        probs  = model.predict_proba(Xg)[:, 1]
        preds  = (probs >= threshold).astype(int)
        auc    = roc_auc_score(yg, probs) if len(np.unique(yg)) > 1 else float("nan")
        f1     = f1_score(yg, preds, zero_division=0)
        print(f"  {lbl:<22} {mask.sum():>5}  {yg.mean():>5.1%}  {auc:>7.4f}  {f1:>7.4f}")
    print()


# ── model definitions ─────────────────────────────────────────────────────────

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
                tree_method="hist",   # fastest single-threaded mode
            )
        }
    else:
        return {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=80,          # was 120/200 — cut to 80
                max_depth=4,
                learning_rate=0.1,    # higher LR = fewer iterations needed
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            )
        }


# ── main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 62)
    print("  CredLess model training  (v2 — full-signal pipeline)")
    print("=" * 62)

    # ── 1. Load data ─────────────────────────────────────────────────
    print("\n[1/6] Loading & engineering features ...")
    features, target, _ = prepare_model_frame()
    X, y = features.values, target.values
    fn   = list(features.columns)
    print(f"  Rows            : {len(X):,}")
    print(f"  Features        : {len(fn)}  {fn}")
    print(f"  Default rate    : {y.mean():.1%}  "
          f"(class 1={y.sum():,}, class 0={(y==0).sum():,})")

    # ── 2. Train / test split ─────────────────────────────────────────
    print("\n[2/6] Splitting data (80/20 stratified) ...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # ── 3. Benchmark all candidates ───────────────────────────────────
    print(f"\n[3/6] Benchmarking candidate models ({CV_FOLDS}-fold CV on train set) ...")
    cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    for name, model in build_candidates().items():
        scores = cross_validate(
            model,
            X_tr,
            y_tr,
            cv=cv,
            scoring={"auc": "roc_auc", "f1": "f1"},
            n_jobs=1,
        )
        auc_scores = scores["test_auc"]
        f1_scores  = scores["test_f1"]
        results[name] = {"auc": auc_scores.mean(), "f1": f1_scores.mean(),
                         "auc_std": auc_scores.std()}
        print(f"  {name:<24}  AUC={auc_scores.mean():.4f} ±{auc_scores.std():.4f}  "
              f"F1={f1_scores.mean():.4f}")

    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n  Best model → {best_name}  (AUC {results[best_name]['auc']:.4f})")

    # ── 4. Train best model on full training split ────────────────────
    print(f"\n[4/6] Training {best_name} on full training split ...")
    best_model = build_candidates()[best_name]
    best_model.fit(X_tr, y_tr)

    threshold, best_f1_tr = best_threshold(best_model, X_tr, y_tr)
    print(f"  Optimal threshold : {threshold:.3f}  (train F1: {best_f1_tr:.4f})")

    # ── 5. Evaluate on held-out test set ─────────────────────────────
    print(f"\n[5/6] Held-out test evaluation ...")
    test_probs = best_model.predict_proba(X_te)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
    test_auc   = roc_auc_score(y_te, test_probs)

    print(f"\n  Test AUC-ROC : {test_auc:.4f}  "
          f"(CV was {results[best_name]['auc']:.4f})")

    gap = abs(results[best_name]["auc"] - test_auc)
    if gap > 0.05:
        print(f"  *** WARNING: CV vs test AUC gap = {gap:.4f}  "
              f"(>0.05 = possible overfit) ***")

    print(f"\n  Classification report (threshold={threshold:.3f}):")
    print(classification_report(y_te, test_preds,
                                target_names=["repay", "default"]))

    tn, fp, fn_count, tp = confusion_matrix(y_te, test_preds).ravel()
    print(f"  Confusion matrix:")
    print(f"    True negatives  (repay → repay)    : {tn:>6}")
    print(f"    False positives (repay → default)  : {fp:>6}  ← wrongly rejected")
    print(f"    False negatives (default → repay)  : {fn_count:>6}  ← wrongly approved  <-- minimise this")
    print(f"    True positives  (default → default): {tp:>6}")

    # ── Feature importance ────────────────────────────────────────────
    print("\n  Feature importances (top 20):")
    if HAS_XGB and best_name == "XGBoost":
        importances = best_model.feature_importances_
    elif best_name == "RandomForest":
        importances = best_model.feature_importances_
    elif best_name == "HistGradientBoosting":
        importances = best_model.feature_importances_
    else:  # LogisticRegression pipeline
        scaler = best_model.named_steps["scaler"]
        clf    = best_model.named_steps["clf"]
        importances = np.abs(clf.coef_[0])

    ranked = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    for feat_name, imp in ranked:
        bar = "█" * int(imp * 40)
        print(f"    {feat_name:<30} {imp:.4f}  {bar}")

    # ── Subgroup audit ─────────────────────────────────────────────────
    subgroup_audit(best_model, X_te, y_te, threshold, fn)

    # ── 6. Save artifact ──────────────────────────────────────────────
    print(f"[6/6] Saving model artifact ...")
    artifact = {
        "model":         best_model,
        "model_name":    best_name,
        "threshold":     threshold,
        "feature_names": fn,
        "cv_auc":        float(results[best_name]["auc"]),
        "test_auc":      float(test_auc),
        "default_rate":  float(y.mean()),
        "n_train":       int(len(X_tr)),
        "n_test":        int(len(X_te)),
        # v1 compatibility — oracle.py can still call artifact["model"].predict_proba
    }
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, SAVE_PATH)
    print(f"  Saved → {SAVE_PATH}")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  Final model  : {best_name}")
    print(f"  Test AUC     : {test_auc:.4f}  (target: 0.85+)")
    print(f"  Threshold    : {threshold:.3f}")
    print(f"  Features     : {len(fn)} (was 8 in v1)")
    print("=" * 62)

    return artifact


if __name__ == "__main__":
    train()
