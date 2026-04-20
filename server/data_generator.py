"""
Dataset-backed applicant generation aligned to the trained 20-feature schema.

The environment now samples real rows from the cleaned CSV and uses the
engineered features produced by `credless_model.dataset_pipeline`. That keeps
the observation space, oracle inputs, and reward logic on the same feature set.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

import numpy as np

from credless_model.dataset_pipeline import FEATURE_NAMES, load_dataset_cache

FIELD_NAMES: List[str] = list(FEATURE_NAMES)
DEFAULT_VISIBLE_FIELDS = [
    "payment_reliability",
    "transaction_health",
    "income_capacity_score",
    "employment_stability",
    "account_maturity",
    "location_risk_index",
]
DIFFICULTY_PARTIALS = {
    "easy": 4,
    "medium": 6,
    "hard": 8,
}
ADVERSARIAL_TARGETS = [
    "income_capacity_score",
    "net_worth_score",
    "payment_reliability",
]
WITHHOLD_TARGETS = [
    "overdraft_risk",
    "medical_stress_score",
    "total_delinquency_score",
    "real_estate_exposure",
]
CONFIDENCE_SENSITIVE_FIELDS = set(ADVERSARIAL_TARGETS + WITHHOLD_TARGETS)


def _dataset_features():
    return load_dataset_cache()["features"]


def _dataset_cleaned():
    return load_dataset_cache()["cleaned"]


def _build_field_ranges() -> Dict[str, tuple]:
    features = _dataset_features()
    ranges: Dict[str, tuple] = {}
    for field in FIELD_NAMES:
        column = features[field].astype(float)
        ranges[field] = (float(column.min()), float(column.max()))
    return ranges


FIELD_RANGES: Dict[str, tuple] = _build_field_ranges()


def _clip_to_range(field: str, value: float) -> float:
    lo, hi = FIELD_RANGES[field]
    return float(np.clip(value, lo, hi))


def _difficulty_to_adversarial_prob(difficulty: str) -> float:
    if difficulty == "hard":
        return 0.8
    if difficulty == "medium":
        return 0.35
    return 0.1


def _sample_row(seed: Optional[int]) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    features = _dataset_features()
    cleaned = _dataset_cleaned()
    idx = int(rng.integers(0, len(features)))
    return {
        "feature_row": features.iloc[idx].astype(float).to_dict(),
        "raw_row": cleaned.iloc[idx].to_dict(),
    }


def _apply_observation_noise(features: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
    presented = dict(features)
    for field in FIELD_NAMES:
        if rng.random() < 0.12:
            presented[field] = _clip_to_range(
                field,
                presented[field] * float(rng.uniform(0.92, 1.08)),
            )
    return presented


def _apply_applicant_behavior(
    true_features: Dict[str, float],
    difficulty: str,
    rng: np.random.Generator,
) -> Dict[str, object]:
    presented = _apply_observation_noise(true_features, rng)
    confidence = {
        field: round(float(rng.uniform(0.72, 0.98)), 3)
        for field in FIELD_NAMES
    }
    fabricated_fields: List[str] = []
    withheld_fields: List[str] = []

    is_adversarial = rng.random() < _difficulty_to_adversarial_prob(difficulty)
    behavior = "adversarial" if is_adversarial else "honest"

    if is_adversarial:
        for field in ADVERSARIAL_TARGETS:
            presented[field] = _clip_to_range(
                field,
                presented[field] + float(rng.uniform(0.05, 0.14)),
            )
            confidence[field] = round(float(rng.uniform(0.38, 0.65)), 3)
            fabricated_fields.append(field)

        hidden_candidates = [field for field in WITHHOLD_TARGETS if field in FIELD_NAMES]
        withheld_fields.extend(hidden_candidates[: int(rng.integers(2, min(4, len(hidden_candidates)) + 1))])
        for field in withheld_fields:
            confidence[field] = round(float(rng.uniform(0.35, 0.60)), 3)

    return {
        "presented_features": presented,
        "confidence": confidence,
        "withheld_fields": sorted(set(withheld_fields)),
        "fabricated_fields": sorted(set(fabricated_fields)),
        "behavior": behavior,
        "is_adversarial": is_adversarial,
    }


def _select_hidden_fields(
    difficulty: str,
    rng: np.random.Generator,
    mandatory_hidden: Optional[List[str]] = None,
) -> List[str]:
    hidden_count = DIFFICULTY_PARTIALS.get(difficulty, 6)
    mandatory_hidden = list(dict.fromkeys(mandatory_hidden or []))
    remaining = [field for field in FIELD_NAMES if field not in mandatory_hidden]
    extra_needed = max(0, hidden_count - len(mandatory_hidden))
    sampled = list(rng.choice(remaining, size=extra_needed, replace=False)) if extra_needed else []
    return sorted(set(mandatory_hidden + sampled))


def generate_applicant(seed: Optional[int] = None, difficulty: str = "easy") -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    sample = _sample_row(seed)
    true_features = sample["feature_row"]
    raw_row = sample["raw_row"]
    behavior = _apply_applicant_behavior(true_features, difficulty, rng)
    hidden_fields = _select_hidden_fields(
        difficulty=difficulty,
        rng=rng,
        mandatory_hidden=behavior["withheld_fields"],
    )

    visible_fields = [field for field in DEFAULT_VISIBLE_FIELDS if field not in hidden_fields]
    if len(visible_fields) < 4:
        for field in FIELD_NAMES:
            if field not in hidden_fields and field not in visible_fields:
                visible_fields.append(field)
            if len(visible_fields) >= 4:
                break

    uncertainty_flags = {
        field: round(1.0 - behavior["confidence"][field], 3)
        for field in FIELD_NAMES
    }
    data_quality = "adversarial" if behavior["is_adversarial"] else "observed_with_noise"

    return {
        "applicant_id": str(uuid.uuid4())[:8].upper(),
        "features": true_features,
        "presented_features": behavior["presented_features"],
        "field_confidence": behavior["confidence"],
        "uncertainty_flags": uncertainty_flags,
        "hidden_fields": hidden_fields,
        "visible_fields": sorted(set(visible_fields)),
        "data_quality": data_quality,
        "applicant_behavior": behavior["behavior"],
        "is_adversarial": behavior["is_adversarial"],
        "fabricated_fields": behavior["fabricated_fields"],
        "withheld_fields": behavior["withheld_fields"],
        "raw_row": raw_row,
        "source": "dataset_sample",
    }
