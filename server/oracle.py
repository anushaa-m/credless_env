import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "credless_model" / "model.pkl"

FEATURE_ORDER = [
    "transaction_activity", "payment_consistency", "account_stability",
    "overdraft_count", "digital_usage", "salary_consistency",
    "failed_tx_ratio", "account_age",
]


class CredLessOracle:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run: python credless_model/train.py"
            )
        artifact = joblib.load(MODEL_PATH)

        if isinstance(artifact, dict):
            self.model = artifact["model"]
            self.model_name = artifact.get("model_name", "unknown")
            self.metrics = artifact.get("metrics", {})
            thresholds = artifact.get("risk_thresholds", {})
            self.low_thresh = thresholds.get("low_risk", 0.40)
            self.high_thresh = thresholds.get("medium_risk", 0.70)
        else:
            self.model = artifact
            self.model_name = "unknown"
            self.metrics = {}
            self.low_thresh = 0.40
            self.high_thresh = 0.70

        auc = self.metrics.get("test_auc", "?")
        auc_display = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        print(
            f"[Oracle] model={self.model_name}  "
            f"test_auc={auc_display}  "
            f"thresholds=({self.low_thresh}, {self.high_thresh})"
        )

    def predict(self, features: dict) -> dict:
        x = np.array([[features[f] for f in FEATURE_ORDER]])
        prob = float(self.model.predict_proba(x)[0][1])

        if prob < self.low_thresh:
            tier, decision = "low_risk", "approve"
        elif prob < self.high_thresh:
            tier, decision = "medium_risk", "approve"
        else:
            tier, decision = "high_risk", "deny"

        return {
            "tier": tier,
            "decision": decision,
            "default_prob": round(prob, 6),
        }
