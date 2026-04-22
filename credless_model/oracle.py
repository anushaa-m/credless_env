from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
METADATA_PATH = ROOT / "metadata.json"
FEATURE_NAMES_PATH = ROOT / "feature_names.txt"


class CreditOracle:
    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"] if isinstance(artifact, dict) else artifact
        self.feature_names = self._load_feature_names()
        self._explainer = None

        if shap is not None:
            try:
                background = np.full((1, len(self.feature_names)), 0.5, dtype=float)
                self._explainer = shap.Explainer(self.model, background)
            except Exception:
                self._explainer = None

    def _load_feature_names(self) -> List[str]:
        if METADATA_PATH.exists():
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            names = metadata.get("feature_names", [])
            if names:
                return [str(name) for name in names]
        if FEATURE_NAMES_PATH.exists():
            return [line.strip() for line in FEATURE_NAMES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        raise FileNotFoundError("Feature names metadata not found for CreditOracle.")

    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        ordered = [float(features.get(name, 0.5)) for name in self.feature_names]
        return np.array([ordered], dtype=float)

    def predict_risk(self, features: Dict[str, float]) -> float:
        vector = self._vectorize(features)
        return float(np.clip(self.model.predict_proba(vector)[0][1], 0.0, 1.0))

    def get_confidence(self, features: Dict[str, float]) -> float:
        vector = self._vectorize(features)
        probabilities = np.asarray(self.model.predict_proba(vector)[0], dtype=float)
        return float(np.clip(probabilities.max(), 0.0, 1.0))

    def get_top_factors(self, features: Dict[str, float], top_k: int = 5) -> List[List[object]]:
        vector = self._vectorize(features)

        if self._explainer is not None:
            try:
                shap_values = self._explainer(vector)
                values = np.asarray(shap_values.values[0], dtype=float)
                return self._format_top_factors(values, top_k=top_k)
            except Exception:
                pass

        if hasattr(self.model, "coef_"):
            coefficients = np.asarray(self.model.coef_[0], dtype=float) * vector[0]
            return self._format_top_factors(coefficients, top_k=top_k)

        if hasattr(self.model, "feature_importances_"):
            importances = np.asarray(self.model.feature_importances_, dtype=float) * np.abs(vector[0] - 0.5)
            return self._format_top_factors(importances, top_k=top_k)

        return [["oracle_risk", round(self.predict_risk(features), 6)]]

    def _format_top_factors(self, scores: np.ndarray, *, top_k: int) -> List[List[object]]:
        pairs = list(zip(self.feature_names, scores.tolist()))
        ranked = sorted(pairs, key=lambda item: abs(float(item[1])), reverse=True)[:top_k]
        return [[str(name), round(float(value), 6)] for name, value in ranked]
