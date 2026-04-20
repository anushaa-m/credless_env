"""
Reward and auditing helpers for the FinVerse investigation workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List

MIN_SCORE = 0.01
MAX_SCORE = 0.99
PROTECTED_TERMS = {
    "religion",
    "caste",
    "gender",
    "marital status",
    "ethnicity",
    "race",
}


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


def _contains_any_reasoning_term(reasoning: str, values: List[str]) -> bool:
    return any(value in reasoning for value in values)


def audit_terminal_action(
    final_action: Dict[str, Any],
    oracle_truth: Dict[str, Any],
    revealed_fields: Dict[str, Dict[str, Any]],
    market_visible: bool,
    fraud_flags: List[str],
) -> Dict[str, float]:
    reasoning = str(final_action.get("reasoning") or "").strip().lower()
    decision = str(final_action.get("decision") or "").strip().lower()

    oracle_alignment = 1.0 if decision == str(oracle_truth.get("decision", "")).lower() else 0.15
    evidence_terms = [field.lower() for field in revealed_fields]
    evidence_hits = sum(
        1 for field in evidence_terms if field in reasoning or field.replace("_", " ") in reasoning
    )
    evidence_score = min(1.0, evidence_hits / max(2, len(evidence_terms) // 3 or 1))
    market_score = 1.0 if (not market_visible or _contains_any_reasoning_term(reasoning, ["market", "rate", "inflation", "recession", "boom"])) else 0.45
    fraud_score = 1.0 if (not fraud_flags or _contains_any_reasoning_term(reasoning, ["fraud", "confidence", "inconsistent", "verification"])) else 0.55
    reasoning_score = 0.20 if len(reasoning) < 30 else min(1.0, 0.35 + evidence_score * 0.35 + market_score * 0.15 + fraud_score * 0.15)

    bias_penalty = 0.0
    for term in PROTECTED_TERMS:
        if term in reasoning:
            bias_penalty += 0.35

    total = max(0.0, min(1.0, 0.65 * oracle_alignment + 0.35 * reasoning_score - bias_penalty))
    return {
        "score": round(total, 4),
        "oracle_alignment": round(oracle_alignment, 4),
        "reasoning_score": round(reasoning_score, 4),
        "bias_penalty": round(bias_penalty, 4),
    }


def evaluate_terminal_action(
    final_action: Dict[str, Any],
    oracle_truth: Dict[str, Any],
    auditor_result: Dict[str, float],
    requests_made: int,
    queried_market: bool,
    fraud_flags: List[str],
    applicant_is_fraudulent: bool,
) -> Dict[str, float]:
    action_type = str(final_action.get("action_type") or "").lower()
    decision = str(final_action.get("decision") or action_type).lower()

    if action_type == "escalate":
        base_score = 0.35
        reward = -0.5
        return {
            "task_score": round(base_score, 4),
            "auditor_score": float(auditor_result.get("score", 0.0)),
            "episode_score": strict_score(0.5 * base_score + 0.5 * float(auditor_result.get("score", 0.0))),
            "reward": reward,
            "efficiency_penalty": 0.0,
            "fraud_bonus": 0.0,
        }

    accuracy = 1.0 if decision == str(oracle_truth.get("decision", "")).lower() else 0.0
    tier_match = 1.0
    if final_action.get("tier"):
        tier_match = 1.0 if str(final_action.get("tier")).lower() == str(oracle_truth.get("tier", "")).lower() else 0.4

    rate_penalty = 0.0
    rate = final_action.get("rate")
    expected_rate = final_action.get("expected_rate")
    if isinstance(rate, (int, float)) and isinstance(expected_rate, (int, float)):
        rate_penalty = min(0.25, abs(float(rate) - float(expected_rate)) / 20.0)

    task_score = max(0.0, 0.75 * accuracy + 0.25 * tier_match - rate_penalty)
    auditor_score = float(auditor_result.get("score", 0.0))

    request_penalty = max(0, requests_made - 3) * 0.08
    market_penalty = 0.05 if not queried_market else 0.0
    fraud_bonus = 0.15 if fraud_flags and applicant_is_fraudulent else 0.0
    false_alarm_penalty = 0.10 if fraud_flags and not applicant_is_fraudulent else 0.0

    episode_score = strict_score(0.65 * task_score + 0.35 * auditor_score)
    reward = round(episode_score - request_penalty - market_penalty + fraud_bonus - false_alarm_penalty, 4)
    return {
        "task_score": round(task_score, 4),
        "auditor_score": round(auditor_score, 4),
        "episode_score": episode_score,
        "reward": reward,
        "efficiency_penalty": round(request_penalty + market_penalty + false_alarm_penalty, 4),
        "fraud_bonus": round(fraud_bonus, 4),
    }
