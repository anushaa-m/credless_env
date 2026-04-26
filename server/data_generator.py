"""
Dataset-backed applicant generation aligned to the trained 20-feature schema.

The environment now samples real rows from the cleaned CSV and uses the
engineered features produced by `credless_model.dataset_pipeline`. That keeps
the observation space, oracle inputs, and reward logic on the same feature set.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
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
    "transaction_health",
]
WITHHOLD_TARGETS = [
    "overdraft_risk",
    "medical_stress_score",
    "total_delinquency_score",
    "real_estate_exposure",
]
CONFIDENCE_SENSITIVE_FIELDS = set(ADVERSARIAL_TARGETS + WITHHOLD_TARGETS)


@dataclass(frozen=True)
class ApplicantAgent:
    deception_level: float = 0.0

    def distort_features(
        self,
        true_features: Dict[str, float],
        rng: np.random.Generator,
    ) -> tuple[Dict[str, float], Dict[str, float], List[str], List[str]]:
        level = float(np.clip(self.deception_level, 0.0, 1.0))
        presented = _apply_observation_noise(true_features, rng)
        confidence = {
            field: round(float(rng.uniform(0.72, 0.98)), 3)
            for field in FIELD_NAMES
        }
        fabricated_fields: List[str] = []
        withheld_fields: List[str] = []

        if level <= 0.0:
            return presented, confidence, fabricated_fields, withheld_fields

        uplift = float(rng.uniform(0.04, 0.10) + (0.08 * level))
        for field in ADVERSARIAL_TARGETS:
            presented[field] = _clip_to_range(field, presented[field] + uplift)
            confidence[field] = round(float(rng.uniform(0.35, 0.70 - 0.20 * level)), 3)
            fabricated_fields.append(field)

        hidden_candidates = [field for field in WITHHOLD_TARGETS if field in FIELD_NAMES]
        max_hidden = min(len(hidden_candidates), 2 + int(round(2 * level)))
        if max_hidden > 0:
            hidden_count = int(rng.integers(1, max_hidden + 1))
            withheld_fields.extend(hidden_candidates[:hidden_count])
        for field in withheld_fields:
            confidence[field] = round(float(rng.uniform(0.30, 0.62 - 0.15 * level)), 3)

        return presented, confidence, fabricated_fields, withheld_fields


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


def _default_deception_level(difficulty: str, rng: np.random.Generator) -> float:
    if difficulty == "hard":
        return float(rng.uniform(0.65, 1.0))
    if difficulty == "medium":
        return float(rng.uniform(0.35, 0.70))
    return float(rng.uniform(0.15, 0.40))


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
    deception_level: Optional[float] = None,
) -> Dict[str, object]:
    is_adversarial = rng.random() < _difficulty_to_adversarial_prob(difficulty)
    behavior = "adversarial" if is_adversarial else "honest"
    effective_deception = (
        float(np.clip(deception_level, 0.0, 1.0))
        if deception_level is not None
        else (_default_deception_level(difficulty, rng) if is_adversarial else 0.0)
    )
    if deception_level is not None:
        is_adversarial = effective_deception > 0.0
        behavior = "adversarial" if is_adversarial else "honest"

    applicant_agent = ApplicantAgent(effective_deception if is_adversarial else 0.0)
    presented, confidence, fabricated_fields, withheld_fields = applicant_agent.distort_features(
        true_features=true_features,
        rng=rng,
    )

    return {
        "presented_features": presented,
        "confidence": confidence,
        "withheld_fields": sorted(set(withheld_fields)),
        "fabricated_fields": sorted(set(fabricated_fields)),
        "behavior": behavior,
        "is_adversarial": is_adversarial,
        "deception_level": round(float(effective_deception if is_adversarial else 0.0), 3),
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


def generate_applicant(
    seed: Optional[int] = None,
    difficulty: str = "easy",
    deception_level: Optional[float] = None,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    sample = _sample_row(seed)
    true_features = sample["feature_row"]
    raw_row = sample["raw_row"]
    behavior = _apply_applicant_behavior(true_features, difficulty, rng, deception_level=deception_level)
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
    data_quality = "self_reported" if behavior["is_adversarial"] else "observed_with_noise"

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
        "deception_level": behavior["deception_level"],
        "fabricated_fields": behavior["fabricated_fields"],
        "withheld_fields": behavior["withheld_fields"],
        "raw_row": raw_row,
        "source": "dataset_sample",
    }
