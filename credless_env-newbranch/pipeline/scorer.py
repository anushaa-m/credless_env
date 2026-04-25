"""
Scoring module for FinVerse compatibility.

This version keeps your weighting logic but normalizes project-native labels:
  - decisions: approve / deny / reject
  - tiers: A/B/C or low_risk/medium_risk/high_risk
"""

from __future__ import annotations

_TIER_ORDER = {"A": 0, "B": 1, "C": 2}
_TIER_ALIASES = {
    "a": "A",
    "b": "B",
    "c": "C",
    "low_risk": "A",
    "medium_risk": "B",
    "high_risk": "C",
}
_DECISION_ALIASES = {
    "approve": "approve",
    "deny": "reject",
    "reject": "reject",
}


def _normalize_decision(value: object) -> str:
    return _DECISION_ALIASES.get(str(value or "").strip().lower(), "reject")


def _normalize_tier(value: object) -> str:
    return _TIER_ALIASES.get(str(value or "").strip().lower(), "C")


def _normalize_confidence(value: object) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.5
    return max(0.0, min(1.0, conf))


def evaluate_prediction(pred: dict, oracle: dict) -> float:
    decision_score = (
        1.0
        if _normalize_decision(pred.get("decision")) == _normalize_decision(oracle.get("decision"))
        else 0.0
    )

    pred_tier = _TIER_ORDER[_normalize_tier(pred.get("risk_tier", pred.get("tier")))]
    oracle_tier = _TIER_ORDER[_normalize_tier(oracle.get("risk_tier", oracle.get("tier")))]
    tier_diff = abs(pred_tier - oracle_tier)
    tier_score = max(0.0, 1.0 - tier_diff * 0.50)

    pred_conf = _normalize_confidence(pred.get("confidence", pred.get("default_prob", 0.5)))
    oracle_conf = _normalize_confidence(oracle.get("confidence", oracle.get("default_prob", 0.5)))
    conf_score = 1.0 - abs(pred_conf - oracle_conf)

    total = (0.60 * decision_score) + (0.25 * tier_score) + (0.15 * conf_score)
    return round(total, 4)


def batch_score(predictions: list[dict], oracles: list[dict]) -> dict:
    assert len(predictions) == len(oracles), "Length mismatch between preds and oracles"

    scores = [evaluate_prediction(pred, oracle) for pred, oracle in zip(predictions, oracles)]
    decision_matches = [
        1
        for pred, oracle in zip(predictions, oracles)
        if _normalize_decision(pred.get("decision")) == _normalize_decision(oracle.get("decision"))
    ]
    tier_matches = [
        1
        for pred, oracle in zip(predictions, oracles)
        if _normalize_tier(pred.get("risk_tier", pred.get("tier")))
        == _normalize_tier(oracle.get("risk_tier", oracle.get("tier")))
    ]

    n = len(scores)
    return {
        "scores": scores,
        "avg_score": round(sum(scores) / n, 4) if n else 0.0,
        "decision_acc": round(len(decision_matches) / n, 4) if n else 0.0,
        "tier_acc": round(len(tier_matches) / n, 4) if n else 0.0,
    }
