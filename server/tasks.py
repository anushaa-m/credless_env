"""
Task registry for the FinVerse investigation workflow.
"""

from .data_generator import FIELD_RANGES

PIPELINE_DESCRIPTION = (
    "Each episode is a deterministic binary credit decision task. "
    "The agent must return exactly one uppercase string: APPROVE or REJECT."
)

TASK_REGISTRY = [
    {
        "name": "binary_decision",
        "difficulty": "easy",
        "description": f"{PIPELINE_DESCRIPTION} Focus on correct approve or deny outcomes.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "input": "APPROVE | REJECT",
        },
    },
    {
        "name": "risk_tiering",
        "difficulty": "medium",
        "description": f"{PIPELINE_DESCRIPTION} Pricing and tier selection matter more heavily.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "input": "APPROVE | REJECT",
        },
    },
    {
        "name": "adaptive_inquiry",
        "difficulty": "hard",
        "description": f"{PIPELINE_DESCRIPTION} Fraud detection and long-horizon evidence gathering matter most.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "input": "APPROVE | REJECT",
        },
    },
]

TASK_NAMES = [t["name"] for t in TASK_REGISTRY]
TASK_DIFFICULTY = {t["name"]: t["difficulty"] for t in TASK_REGISTRY}
