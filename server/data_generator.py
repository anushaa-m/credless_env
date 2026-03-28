# server/data_generator.py
"""
Generates synthetic applicant profiles for each episode.
Distributions are calibrated to produce ~30% default rate,
matching the CredLess training data characteristics.
"""
import uuid
import numpy as np
from typing import Dict, Optional

# Feature name → (min, max) for uniform sampling
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

    features = {
        field: float(rng.uniform(lo, hi))
        for field, (lo, hi) in FIELD_RANGES.items()
    }

    return {
        "applicant_id": applicant_id,
        "features":     features,
    }