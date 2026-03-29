# server/oracle.py
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "credless_model" / "model.pkl"

# Exact order must match what train.py uses — do NOT change this list
FEATURE_ORDER = [
    "transaction_activity",
    "payment_consistency",
    "account_stability",
    "overdraft_count",
    "digital_usage",
    "salary_consistency",
    "failed_tx_ratio",
    "account_age",
]


class CredLessOracle:
    """
    Wraps the trained CredLess Logistic Regression model.
    Acts as the ground-truth scorer for all environment tasks.
    Loaded once at server startup and reused across episodes.
    """

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python credless_model/train.py` first."
            )
        self.model = joblib.load(MODEL_PATH)

    def predict(self, features: dict) -> dict:
        """
        Run inference on a feature dict.

        Returns:
            {
              "tier":         "low_risk" | "medium_risk" | "high_risk",
              "decision":     "approve" | "deny",
              "default_prob": float   # P(default) — 0.0 to 1.0
            }
        """
        x = np.array([[features[f] for f in FEATURE_ORDER]])
        prob = float(self.model.predict_proba(x)[0][1])  # P(default)

        if prob < 0.30:
            tier     = "low_risk"
            decision = "approve"
        elif prob < 0.60:
            tier     = "medium_risk"
            decision = "approve"   # would be manual review in prod
        else:
            tier     = "high_risk"
            decision = "deny"

        return {
            "tier":         tier,
            "decision":     decision,
            "default_prob": round(prob, 6),
        }