# server/graders.py
"""
Deterministic graders for each task.
All return float in [0.0, 1.0].
These are the ground-truth scorers used by the /grader endpoint
and reported as episode_score in the final observation.
"""


def grade_binary_decision(
    agent_decision: str,
    oracle_decision: str,
) -> float:
    """
    Easy task grader.
    Full credit (1.0) for correct approve/deny, zero otherwise.
    Completely deterministic — no partial credit.
    """
    return 1.0 if agent_decision.strip().lower() == oracle_decision.strip().lower() else 0.0


def grade_risk_tiering(
    agent_tier: str,
    oracle_tier: str,
    agent_limit: float,
    oracle_default_prob: float,
) -> float:
    """
    Medium task grader.
    60% weight → tier accuracy (full/partial/zero)
    40% weight → credit limit reasonableness vs oracle probability

    Tier scoring:
        Exact match     → 1.0
        Off by one tier → 0.5
        Off by two tiers→ 0.0

    Limit scoring:
        Expected limit = (1 - default_prob) * 500_000 INR
        Penalised by proportional deviation, capped at 0.
    """
    TIER_ORDER = {"low_risk": 0, "medium_risk": 1, "high_risk": 2}

    agent_idx  = TIER_ORDER.get(agent_tier.strip().lower(),  -99)
    oracle_idx = TIER_ORDER.get(oracle_tier.strip().lower(), -99)
    diff       = abs(agent_idx - oracle_idx)

    if diff == 0:
        tier_score = 1.0
    elif diff == 1:
        tier_score = 0.5
    else:
        tier_score = 0.0

    # Credit limit score
    expected_limit = max(10_000.0, (1.0 - oracle_default_prob) * 500_000.0)
    if expected_limit > 0:
        proportional_error = abs(agent_limit - expected_limit) / expected_limit
        limit_score = max(0.0, 1.0 - proportional_error)
    else:
        limit_score = 0.0

    final = round(0.6 * tier_score + 0.4 * limit_score, 4)
    return final


def grade_adaptive_inquiry(
    agent_decision: str,
    oracle_decision: str,
    n_fields_requested: int,
    free_requests: int = 3,   # requests before efficiency penalty starts
) -> float:
    """
    Hard task grader.
    70% weight → correctness (binary)
    30% weight → efficiency (penalised 0.10 per request beyond free_requests)

    Example:
        Correct decision, 2 requests → 0.70 + 0.30 = 1.00
        Correct decision, 5 requests → 0.70 + max(0, 0.30 - 0.20) = 0.80
        Wrong decision,   2 requests → 0.00 + 0.30 = 0.30
    """
    correctness = (
        1.0 if agent_decision.strip().lower() == oracle_decision.strip().lower()
        else 0.0
    )

    over_budget = max(0, n_fields_requested - free_requests)
    efficiency  = max(0.0, 1.0 - 0.10 * over_budget)

    return round(0.7 * correctness + 0.3 * efficiency, 4)