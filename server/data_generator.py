# server/data_generator.py
"""
Generates synthetic applicant profiles for each episode.

The original version sampled every feature independently from a uniform range.
That made applicants unrealistically random and easy to game with a hand-built
rule that approximated the oracle. This version produces correlated features
from a few latent behavioural factors so profiles look more realistic and the
answer is less directly recoverable from a simple shortcut.
"""
import uuid
import numpy as np
from typing import Dict, Optional

# Feature name -> (min, max) for uniform sampling
FIELD_RANGES: Dict[str, tuple] = {
    "transaction_activity":  (0.0,  1.0),   # normalised tx count score
    "payment_consistency":   (0.0,  1.0),   # on-time payment ratio
    "account_stability":     (0.0,  1.0),   # balance stability score
    "overdraft_count":       (0.0, 20.0),   # raw count in last 12 months
    "digital_usage":         (0.0,  1.0),   # UPI/digital payment ratio
    "salary_consistency":    (0.0,  1.0),   # salary credit regularity score
    "failed_tx_ratio":       (0.0,  0.5),   # failed / total transactions
    "account_age":           (1.0, 120.0),  # months
}

# First 4 shown at episode start for adaptive_inquiry task
ALWAYS_VISIBLE = [
    "transaction_activity",
    "payment_consistency",
    "account_stability",
    "overdraft_count",
]

# Last 4 must be requested in adaptive_inquiry task
HIDDEN_INITIALLY = [
    "digital_usage",
    "salary_consistency",
    "failed_tx_ratio",
    "account_age",
]


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _beta_scaled(rng: np.random.Generator, lo: float, hi: float, a: float, b: float) -> float:
    return float(lo + (hi - lo) * rng.beta(a, b))


def generate_applicant(seed: Optional[int] = None) -> Dict:
    """
    Returns a dict:
        {
            "applicant_id": str,
            "features":     {field_name: float, ...}
        }
    """
    rng = np.random.default_rng(seed)
    applicant_id = str(uuid.uuid4())[:8].upper()

    # Shared latent traits create realistic cross-feature relationships.
    reliability = rng.beta(4.2, 2.0)          # higher = steadier repayment behaviour
    liquidity = rng.beta(3.6, 2.4)            # higher = stronger cash-flow buffer
    digital_affinity = rng.beta(3.0, 2.2)     # higher = more digital usage
    volatility = rng.beta(2.1, 4.0)           # higher = noisier / less stable behaviour

    account_age = _clip(
        6.0 + 104.0 * (0.55 * reliability + 0.25 * liquidity + 0.20 * rng.random()),
        *FIELD_RANGES["account_age"],
    )

    payment_consistency = _clip(
        0.18
        + 0.54 * reliability
        + 0.14 * liquidity
        - 0.22 * volatility
        + rng.normal(0.0, 0.06),
        *FIELD_RANGES["payment_consistency"],
    )

    salary_consistency = _clip(
        0.10
        + 0.52 * reliability
        + 0.24 * liquidity
        - 0.16 * volatility
        + 0.06 * np.tanh((account_age - 24.0) / 36.0)
        + rng.normal(0.0, 0.07),
        *FIELD_RANGES["salary_consistency"],
    )

    account_stability = _clip(
        0.12
        + 0.38 * liquidity
        + 0.24 * reliability
        - 0.18 * volatility
        + 0.10 * salary_consistency
        + rng.normal(0.0, 0.06),
        *FIELD_RANGES["account_stability"],
    )

    digital_usage = _clip(
        0.06
        + 0.60 * digital_affinity
        + 0.12 * reliability
        + 0.08 * np.tanh((account_age - 12.0) / 30.0)
        + rng.normal(0.0, 0.08),
        *FIELD_RANGES["digital_usage"],
    )

    transaction_activity = _clip(
        0.08
        + 0.32 * digital_usage
        + 0.18 * liquidity
        + 0.18 * reliability
        + 0.10 * _beta_scaled(rng, 0.0, 1.0, 2.5, 2.2)
        - 0.08 * volatility
        + rng.normal(0.0, 0.07),
        *FIELD_RANGES["transaction_activity"],
    )

    overdraft_count = _clip(
        1.0
        + 12.5 * (1.0 - liquidity)
        + 4.5 * volatility
        + 2.5 * (1.0 - payment_consistency)
        + rng.normal(0.0, 1.8),
        *FIELD_RANGES["overdraft_count"],
    )

    failed_tx_ratio = _clip(
        0.03
        + 0.14 * volatility
        + 0.08 * (1.0 - digital_usage)
        + 0.05 * (overdraft_count / FIELD_RANGES["overdraft_count"][1])
        + rng.normal(0.0, 0.025),
        *FIELD_RANGES["failed_tx_ratio"],
    )

    features = {
        "transaction_activity": transaction_activity,
        "payment_consistency": payment_consistency,
        "account_stability": account_stability,
        "overdraft_count": overdraft_count,
        "digital_usage": digital_usage,
        "salary_consistency": salary_consistency,
        "failed_tx_ratio": failed_tx_ratio,
        "account_age": account_age,
    }

    return {
        "applicant_id": applicant_id,
        "features":     features,
    }
