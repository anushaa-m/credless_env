"""
Task registry for the FinVerse investigation workflow.
"""

from .data_generator import FIELD_RANGES

PIPELINE_DESCRIPTION = (
    "Each episode is a stateful lending investigation. The agent can request additional "
    "applicant fields, reveal market conditions, raise fraud flags, and must finish with "
    "approve, deny, or escalate plus reasoning."
)

TASK_REGISTRY = [
    {
        "name": "binary_decision",
        "difficulty": "easy",
        "description": f"{PIPELINE_DESCRIPTION} Focus on correct approve or deny outcomes.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "request_info": {"action_type": "request_info", "params": {"field": f"one of {list(FIELD_RANGES.keys())}"}},
            "query_market": {"action_type": "query_market", "params": {}},
            "flag_fraud": {"action_type": "flag_fraud", "params": {"reason": "string"}},
            "approve": {"action_type": "approve", "params": {"tier": "optional", "rate": "optional float"}, "reasoning": "string"},
            "deny": {"action_type": "deny", "params": {"tier": "optional", "rate": "optional float"}, "reasoning": "string"},
            "escalate": {"action_type": "escalate", "params": {"review_note": "optional"}, "reasoning": "string"},
            "simple_binary_input": "APPROVE | REJECT",
        },
    },
    {
        "name": "risk_tiering",
        "difficulty": "medium",
        "description": f"{PIPELINE_DESCRIPTION} Pricing and tier selection matter more heavily.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "request_info": {"action_type": "request_info", "params": {"field": f"one of {list(FIELD_RANGES.keys())}"}},
            "query_market": {"action_type": "query_market", "params": {}},
            "flag_fraud": {"action_type": "flag_fraud", "params": {"reason": "string"}},
            "approve": {"action_type": "approve", "params": {"tier": "low_risk | medium_risk | high_risk", "rate": "float"}, "reasoning": "string"},
            "deny": {"action_type": "deny", "params": {"tier": "optional", "rate": "optional float"}, "reasoning": "string"},
            "escalate": {"action_type": "escalate", "params": {"review_note": "optional"}, "reasoning": "string"},
            "simple_binary_input": "APPROVE | REJECT",
        },
    },
    {
        "name": "adaptive_inquiry",
        "difficulty": "hard",
        "description": f"{PIPELINE_DESCRIPTION} Fraud detection and long-horizon evidence gathering matter most.",
        "max_steps": 8,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "request_info": {"action_type": "request_info", "params": {"field": f"one of {list(FIELD_RANGES.keys())}"}},
            "query_market": {"action_type": "query_market", "params": {}},
            "flag_fraud": {"action_type": "flag_fraud", "params": {"reason": "string"}},
            "approve": {"action_type": "approve", "params": {"tier": "optional", "rate": "optional float"}, "reasoning": "string"},
            "deny": {"action_type": "deny", "params": {"tier": "optional", "rate": "optional float"}, "reasoning": "string"},
            "escalate": {"action_type": "escalate", "params": {"review_note": "optional"}, "reasoning": "string"},
            "simple_binary_input": "APPROVE | REJECT",
        },
    },
]

TASK_NAMES = [t["name"] for t in TASK_REGISTRY]
TASK_DIFFICULTY = {t["name"]: t["difficulty"] for t in TASK_REGISTRY}
