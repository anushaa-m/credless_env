# server/tasks.py
"""
Task registry — single source of truth for task metadata.
Consumed by /tasks endpoint and environment validation.
"""
from server.data_generator import FIELD_RANGES

TASK_REGISTRY = [
    {
        "name":        "binary_decision",
        "difficulty":  "easy",
        "description": (
            "The full applicant profile (all 8 features) is provided. "
            "The agent must approve or deny the loan in a single action. "
            "Grader: binary correct/incorrect (0.0 or 1.0)."
        ),
        "max_steps":   2,
        "action_schema": {
            "action_type": "approve | deny",
            "decision":    "approve | deny",
        },
    },
    {
        "name":        "risk_tiering",
        "difficulty":  "medium",
        "description": (
            "Full profile provided. The agent must assign a risk tier AND "
            "suggest a credit limit in INR in a single action. "
            "Grader: tier accuracy (60%) + limit reasonableness (40%)."
        ),
        "max_steps":   2,
        "action_schema": {
            "action_type":  "assign_tier",
            "tier":         "low_risk | medium_risk | high_risk",
            "credit_limit": "float in INR (e.g. 75000.0)",
        },
    },
    {
        "name":        "adaptive_inquiry",
        "difficulty":  "hard",
        "description": (
            "Only 4 of 8 features are shown initially. The agent may request "
            "additional fields (penalised per extra request beyond 3 free), "
            "then must issue an approve/deny decision. "
            "Grader: correctness (70%) + inquiry efficiency (30%)."
        ),
        "max_steps":   12,
        "free_requests": 3,
        "available_fields": list(FIELD_RANGES.keys()),
        "action_schema": {
            "step_1_to_N": {
                "action_type": "request_field",
                "field_name":  f"one of {list(FIELD_RANGES.keys())}",
            },
            "final_step": {
                "action_type": "approve | deny",
                "decision":    "approve | deny",
            },
        },
    },
]

TASK_NAMES = [t["name"] for t in TASK_REGISTRY]