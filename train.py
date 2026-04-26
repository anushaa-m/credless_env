from __future__ import annotations

from pathlib import Path

import joblib

from credless_model.train import train


ARTIFACT_PATH = Path(__file__).resolve().parent / "credless_model" / "model.pkl"


def _persist_oracle_thresholds() -> None:
    if not ARTIFACT_PATH.exists():
        return

    payload = joblib.load(ARTIFACT_PATH)
    if not isinstance(payload, dict):
        return

    threshold = payload.get("threshold")
    if threshold is None:
        return

    # Backward-compatible safeguard: ensure oracle tier thresholds are persisted.
    low_risk_threshold = max(0.10, min(float(threshold) - 0.05, float(threshold) * 0.60))
    medium_risk_threshold = max(low_risk_threshold + 0.05, min(0.90, float(threshold)))
    payload["risk_thresholds"] = {
        "low_risk": round(low_risk_threshold, 4),
        "medium_risk": round(medium_risk_threshold, 4),
    }
    joblib.dump(payload, ARTIFACT_PATH)


def main() -> None:
    train()
    _persist_oracle_thresholds()


if __name__ == "__main__":
    main()
