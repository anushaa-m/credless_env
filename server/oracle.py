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
        self._warned_legacy_fallback = False

        auc = self.metrics.get("test_auc", "?")
        auc_display = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        print(
            f"[Oracle] model={self.model_name}  "
            f"test_auc={auc_display}  "
            f"thresholds=({self.low_thresh}, {self.high_thresh})"
        )

    def _legacy_default_prob(self, features: dict) -> float:
        """
        Fallback scorer for the original 8-feature environment.

        This keeps the environment operational even if the saved model artifact
        was trained on a newer 20-feature dataset pipeline. The score is a
        conservative business-rule approximation: weak stability, high
        overdrafts, and high failed-transaction ratios push applicants toward
        denial faster than the LLM's naive "approve" tendency.
        """
        transaction_activity = float(features["transaction_activity"])
        payment_consistency = float(features["payment_consistency"])
        account_stability = float(features["account_stability"])
        overdraft_count = float(features["overdraft_count"])
        digital_usage = float(features["digital_usage"])
        salary_consistency = float(features["salary_consistency"])
        failed_tx_ratio = float(features["failed_tx_ratio"])
        account_age = float(features["account_age"])

        overdraft_risk = min(max(overdraft_count / 20.0, 0.0), 1.0)
        age_risk = min(max(1.0 - (account_age / 120.0), 0.0), 1.0)
        failed_tx_risk = min(max(failed_tx_ratio / 0.5, 0.0), 1.0)

        risk_score = (
            0.16 * (1.0 - transaction_activity)
            + 0.20 * (1.0 - payment_consistency)
            + 0.18 * (1.0 - account_stability)
            + 0.14 * overdraft_risk
            + 0.06 * (1.0 - digital_usage)
            + 0.12 * (1.0 - salary_consistency)
            + 0.10 * failed_tx_risk
            + 0.04 * age_risk
        )

        # Hard-risk overrides for obviously weak profiles.
        if account_stability < 0.10:
            risk_score += 0.12
        if failed_tx_ratio > 0.35:
            risk_score += 0.10
        if overdraft_count > 12:
            risk_score += 0.08
        if payment_consistency < 0.35 and salary_consistency < 0.35:
            risk_score += 0.08

        return float(np.clip(risk_score, 0.0, 1.0))

    def predict(self, features: dict) -> dict:
        x = np.array([[features[f] for f in FEATURE_ORDER]])
        try:
            prob = float(self.model.predict_proba(x)[0][1])
        except Exception as exc:
            if not self._warned_legacy_fallback:
                print(f"[Oracle] falling back to legacy scorer: {exc}")
                self._warned_legacy_fallback = True
            prob = self._legacy_default_prob(features)

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
