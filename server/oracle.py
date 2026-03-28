import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "credless_model" / "model.pkl"

class CredLessOracle:
    """Wraps the trained CredLess Logistic Regression as ground-truth."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        # Feature order must match training exactly
        self.feature_order = [
            "transaction_activity", "payment_consistency", "account_stability",
            "overdraft_count", "digital_usage", "salary_consistency",
            "failed_tx_ratio", "account_age"
        ]

    def predict(self, features: dict) -> dict:
        """Returns tier, decision, and probability."""
        x = np.array([[features[f] for f in self.feature_order]])
        prob = self.model.predict_proba(x)[0][1]  # P(default)

        if prob < 0.3:
            tier = "low_risk"
            decision = "approve"
        elif prob < 0.6:
            tier = "medium_risk"
            decision = "approve"  # manual review in prod, approve here
        else:
            tier = "high_risk"
            decision = "deny"

        return {"tier": tier, "decision": decision, "default_prob": float(prob)}