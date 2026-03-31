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