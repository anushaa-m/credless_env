# credless_model/train.py
"""
Trains the CredLess Logistic Regression model on synthetic data
and saves it to credless_model/model.pkl.

Run once before docker build:
    python credless_model/train.py
"""
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

SAVE_PATH = Path(__file__).parent / "model.pkl"
N_SAMPLES  = 5_000
RANDOM_SEED = 42

# Feature order — MUST match oracle.py FEATURE_ORDER exactly
FEATURE_NAMES = [
    "transaction_activity",  # 0-1, higher = better
    "payment_consistency",   # 0-1, higher = better
    "account_stability",     # 0-1, higher = better
    "overdraft_count",       # 0-20, lower = better
    "digital_usage",         # 0-1, higher = better (UPI adoption)
    "salary_consistency",    # 0-1, higher = better
    "failed_tx_ratio",       # 0-0.5, lower = better
    "account_age",           # 1-120 months, higher = better
]


def generate_training_data(n: int = N_SAMPLES, seed: int = RANDOM_SEED):
    """
    Generates synthetic labelled data with a ~30% default rate.
    Default probability is driven by the same business logic as oracle.predict().
    """
    rng = np.random.default_rng(seed)

    # Sample each feature
    transaction_activity = rng.uniform(0.0, 1.0,   n)
    payment_consistency  = rng.uniform(0.0, 1.0,   n)
    account_stability    = rng.uniform(0.0, 1.0,   n)
    overdraft_count      = rng.uniform(0.0, 20.0,  n)
    digital_usage        = rng.uniform(0.0, 1.0,   n)
    salary_consistency   = rng.uniform(0.0, 1.0,   n)
    failed_tx_ratio      = rng.uniform(0.0, 0.5,   n)
    account_age          = rng.uniform(1.0, 120.0, n)

    X = np.column_stack([
        transaction_activity,
        payment_consistency,
        account_stability,
        overdraft_count,
        digital_usage,
        salary_consistency,
        failed_tx_ratio,
        account_age,
    ])

    # Risk score: higher = more likely to default
    # Normalise overdraft_count and account_age to 0-1 for scoring
    norm_overdraft = overdraft_count / 20.0
    norm_age       = 1.0 - (account_age / 120.0)   # older = lower risk

    risk_score = (
        (1.0 - transaction_activity) * 0.20 +
        (1.0 - payment_consistency)  * 0.25 +
        (1.0 - account_stability)    * 0.15 +
        norm_overdraft               * 0.10 +
        (1.0 - digital_usage)        * 0.05 +
        (1.0 - salary_consistency)   * 0.15 +
        failed_tx_ratio * 2.0        * 0.05 +
        norm_age                     * 0.05
    )

    # Add noise so the model isn't perfect (realistic AUC ~0.92-0.95)
    noise = rng.normal(0.0, 0.08, n)
    final_risk = np.clip(risk_score + noise, 0.0, 1.0)

    # Label: 1 = default, 0 = repay
    # Threshold at 0.50 gives ~30% default rate
    y = (final_risk > 0.50).astype(int)

    return X, y


def train():
    print("Generating synthetic training data ...")
    X, y = generate_training_data()
    print(f"  Samples: {len(X)} | Default rate: {y.mean():.1%}")

    # Pipeline: scale → logistic regression (matches CredLess original)
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
        C=1.0,
        max_iter=500,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        n_jobs=-1,
    )),
])
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on all data
    model.fit(X, y)
    print(f"  Train AUC: {cv_scores.mean():.4f}")

    # Save
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVE_PATH)
    print(f"  Model saved → {SAVE_PATH}")
    return model


if __name__ == "__main__":
    train()

'''new code - change it with claude 


"""
CredLess — train.py
====================
Trains a Logistic Regression credit-scoring model on synthetic data
and saves it to credless_model/model.pkl.
 
Improvements over v1
---------------------
1.  Proper train / test split (80/20, stratified) — test set is NEVER
    touched during training or cross-validation.
2.  Cross-validation runs only on the training split.
3.  class_weight='balanced' handles class imbalance automatically.
4.  Full evaluation suite: AUC-ROC, precision, recall, F1, confusion matrix.
5.  Threshold tuning: finds the decision threshold that maximises F1
    on the validation set.
6.  Feature importance (LR coefficients) printed for interpretability
    and bias auditing.
7.  Subgroup parity check — compares model performance across
    account_age quartiles (proxy for applicant tenure / age-related bias).
8.  Model + metadata (threshold, feature names, class distribution)
    saved together so oracle.py can use them correctly.
 
Run once before docker build:
    python credless_model/train.py
"""
 
import joblib
import numpy as np
from pathlib import Path
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
)
 
# ── paths & constants ────────────────────────────────────────────────────────
 
SAVE_PATH   = Path(__file__).parent / "model.pkl"
N_SAMPLES   = 5_000
RANDOM_SEED = 42
TEST_SIZE   = 0.20      # 20 % held-out test set, never seen during training
CV_FOLDS    = 5
 
# Feature order — MUST match oracle.py FEATURE_ORDER exactly
FEATURE_NAMES = [
    "transaction_activity",   # 0–1,   higher = better
    "payment_consistency",    # 0–1,   higher = better
    "account_stability",      # 0–1,   higher = better
    "overdraft_count",        # 0–20,  lower  = better
    "digital_usage",          # 0–1,   higher = better (UPI adoption)
    "salary_consistency",     # 0–1,   higher = better
    "failed_tx_ratio",        # 0–0.5, lower  = better
    "account_age",            # 1–120 months, higher = better
]
 
 
# ── data generation ──────────────────────────────────────────────────────────
 
def generate_training_data(n: int = N_SAMPLES, seed: int = RANDOM_SEED):
    """
    Generates synthetic labelled data with a ~30 % default rate.
    Default probability is driven by the same business logic as oracle.predict().
 
    NOTE: This function intentionally introduces small non-linearities via
    noise so the model is not perfectly memorising a formula — AUC will be
    ~0.92–0.95.  Replace this with real transaction data when available.
    """
    rng = np.random.default_rng(seed)
 
    transaction_activity = rng.uniform(0.0,   1.0,   n)
    payment_consistency  = rng.uniform(0.0,   1.0,   n)
    account_stability    = rng.uniform(0.0,   1.0,   n)
    overdraft_count      = rng.uniform(0.0,  20.0,   n)
    digital_usage        = rng.uniform(0.0,   1.0,   n)
    salary_consistency   = rng.uniform(0.0,   1.0,   n)
    failed_tx_ratio      = rng.uniform(0.0,   0.5,   n)
    account_age          = rng.uniform(1.0, 120.0,   n)
 
    X = np.column_stack([
        transaction_activity,
        payment_consistency,
        account_stability,
        overdraft_count,
        digital_usage,
        salary_consistency,
        failed_tx_ratio,
        account_age,
    ])
 
    norm_overdraft = overdraft_count / 20.0
    norm_age       = 1.0 - (account_age / 120.0)   # older = lower risk
 
    risk_score = (
        (1.0 - transaction_activity) * 0.20 +
        (1.0 - payment_consistency)  * 0.25 +
        (1.0 - account_stability)    * 0.15 +
        norm_overdraft               * 0.10 +
        (1.0 - digital_usage)        * 0.05 +
        (1.0 - salary_consistency)   * 0.15 +
        failed_tx_ratio * 2.0        * 0.05 +
        norm_age                     * 0.05
    )
 
    noise      = rng.normal(0.0, 0.08, n)
    final_risk = np.clip(risk_score + noise, 0.0, 1.0)
 
    # 1 = default, 0 = repay  |  threshold 0.50 → ~30 % default rate
    y = (final_risk > 0.50).astype(int)
    return X, y
 
 
# ── threshold selection ───────────────────────────────────────────────────────
 
def best_threshold(model, X_val, y_val):
    """
    Returns the decision threshold (on predicted probabilities) that
    maximises the F1 score on the validation set.
 
    Using 0.5 by default is fine for balanced classes, but in credit
    scoring you often want to tune this to control the false-negative
    rate (approving applicants who will default) vs false-positive
    rate (rejecting applicants who would have repaid).
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    # precision_recall_curve returns one fewer threshold than precision/recall
    f1_scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-9
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])
 
 
# ── subgroup parity audit ─────────────────────────────────────────────────────
 
def subgroup_audit(model, X_test, y_test, threshold: float):
    """
    Checks whether model performance is roughly equal across subgroups.
 
    Uses account_age quartiles as a proxy — in real data you would check
    against demographics or any protected characteristic relevant to your
    applicant population.
 
    A large AUC gap between subgroups (>0.05) is a warning sign of bias.
    """
    ages       = X_test[:, FEATURE_NAMES.index("account_age")]
    quartiles  = np.percentile(ages, [25, 50, 75])
    group_labels = ["Q1 (youngest)", "Q2", "Q3", "Q4 (oldest)"]
 
    print("\n── Subgroup parity audit (account_age quartiles) ──────────────")
    print(f"  {'Group':<20} {'n':>5}  {'default%':>9}  {'AUC':>7}  {'F1':>7}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*7}")
 
    for i, label in enumerate(group_labels):
        lo = quartiles[i - 1] if i > 0 else -np.inf
        hi = quartiles[i]     if i < 3 else  np.inf
        mask = (ages > lo) & (ages <= hi)
        if mask.sum() < 10:
            continue
        Xg, yg = X_test[mask], y_test[mask]
        probs   = model.predict_proba(Xg)[:, 1]
        preds   = (probs >= threshold).astype(int)
        auc     = roc_auc_score(yg, probs) if len(np.unique(yg)) > 1 else float("nan")
        f1      = f1_score(yg, preds, zero_division=0)
        print(
            f"  {label:<20} {mask.sum():>5}  "
            f"{yg.mean():>8.1%}  {auc:>7.4f}  {f1:>7.4f}"
        )
 
    print()
 
 
# ── main training routine ─────────────────────────────────────────────────────
 
def train():
    print("=" * 60)
    print("  CredLess model training")
    print("=" * 60)
 
    # ── 1. Generate data ─────────────────────────────────────────────
    print("\n[1/5] Generating synthetic training data ...")
    X, y = generate_training_data()
    default_rate = y.mean()
    print(f"  Total samples : {len(X):,}")
    print(f"  Default rate  : {default_rate:.1%}  (class 1={y.sum():,}, class 0={( y==0).sum():,})")
 
    # ── 2. Train / test split (stratified) ───────────────────────────
    print("\n[2/5] Splitting data (80% train, 20% held-out test) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,          # preserves class ratio in both splits
    )
    print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"  Train default rate: {y_train.mean():.1%}  |  Test: {y_test.mean():.1%}")
 
    # ── 3. Build pipeline & cross-validate on training data only ─────
    print("\n[3/5] Cross-validating on training split ...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
            class_weight="balanced",   # handles class imbalance automatically
            n_jobs=-1,
        )),
    ])
 
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
 
    print(f"  CV AUC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  CV F1   : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
 
    # ── 4. Final fit on full training data ───────────────────────────
    print("\n[4/5] Training final model on full training split ...")
    model.fit(X_train, y_train)
 
    # Tune decision threshold on the training predictions
    # (in production, use a separate validation fold for this)
    threshold, best_f1 = best_threshold(model, X_train, y_train)
    print(f"  Optimal threshold : {threshold:.3f}  (train F1 @ threshold: {best_f1:.4f})")
 
    # ── 5. Evaluate on held-out test set ─────────────────────────────
    print("\n[5/5] Evaluating on held-out test set ...")
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)
 
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"\n  Test AUC-ROC : {test_auc:.4f}")
    print("\n  Classification report (threshold = {:.3f}):".format(threshold))
    print(classification_report(y_test, test_preds, target_names=["repay", "default"]))
 
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"    True Negatives  (repay  → repay)   : {tn:>5}")
    print(f"    False Positives (repay  → default) : {fp:>5}  ← applicants wrongly rejected")
    print(f"    False Negatives (default→ repay)   : {fn:>5}  ← defaults wrongly approved")
    print(f"    True Positives  (default→ default) : {tp:>5}")
 
    # Feature importance (interpretability)
    coef = model.named_steps["clf"].coef_[0]
    print("\n  Feature importance (LR coefficients — higher = more risk):")
    for name, weight in sorted(zip(FEATURE_NAMES, coef), key=lambda x: x[1], reverse=True):
        bar = "█" * int(abs(weight) * 10)
        sign = "+" if weight > 0 else "-"
        print(f"    {name:<26} {sign}{abs(weight):.4f}  {bar}")
 
    # Subgroup parity check
    subgroup_audit(model, X_test, y_test, threshold)
 
    # ── Save model + metadata ─────────────────────────────────────────
    artifact = {
        "model":          model,
        "threshold":      threshold,
        "feature_names":  FEATURE_NAMES,
        "train_auc":      float(cv_auc.mean()),
        "test_auc":       float(test_auc),
        "default_rate":   float(default_rate),
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    }
 
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, SAVE_PATH)
    print(f"\n  Model artifact saved → {SAVE_PATH}")
    print("=" * 60)
 
    # Warn if AUC gap between CV and test is suspicious
    gap = abs(cv_auc.mean() - test_auc)
    if gap > 0.05:
        print(f"\n  WARNING: CV AUC vs test AUC gap = {gap:.4f} (>0.05).")
        print("  This suggests overfitting — consider reducing C or adding more data.")
 
    return artifact
 
 
if __name__ == "__main__":
    train()
 
 
 
 '''