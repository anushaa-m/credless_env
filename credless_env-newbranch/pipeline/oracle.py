"""
FinVerse-compatible oracle wrapper.

This exposes the deterministic raw-row oracle in the interface expected by the
compatibility pipeline:
  - decision: approve / reject
  - risk_tier: A / B / C
  - confidence: 0..1
"""

from __future__ import annotations

from typing import Dict

from server.oracle import (
    RAW_TIER_A_THRESHOLD as TIER_A_THRESHOLD,
    RAW_TIER_B_THRESHOLD as TIER_B_THRESHOLD,
    oracle_decision as _server_oracle_decision,
)


def score_to_tier(confidence: float) -> str:
    if float(confidence) >= TIER_A_THRESHOLD:
        return "A"
    if float(confidence) >= TIER_B_THRESHOLD:
        return "B"
    return "C"


def score_to_decision(confidence: float) -> str:
    return "approve" if score_to_tier(confidence) in {"A", "B"} else "reject"


def score_to_prediction(confidence: float) -> Dict[str, object]:
    confidence = max(0.0, min(1.0, float(confidence)))
    return {
        "decision": score_to_decision(confidence),
        "risk_tier": score_to_tier(confidence),
        "confidence": confidence,
    }


def oracle_decision(applicant: Dict[str, object]) -> Dict[str, object]:
    result = _server_oracle_decision(applicant)
    return {
        "decision": "reject" if result["decision"] == "deny" else result["decision"],
        "risk_tier": result["risk_tier"],
        "confidence": float(result["confidence"]),
    }
