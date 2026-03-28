# models.py
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

# ── Action ────────────────────────────────────────────────────────────────────
class CreditAction(Action):
    """
    action_type : "request_field" | "approve" | "deny" | "assign_tier"
    field_name  : required when action_type == "request_field"
    decision    : "approve" | "deny"
    tier        : "low_risk" | "medium_risk" | "high_risk"
    credit_limit: float in INR (for risk_tiering task)
    """
    action_type:   str            = Field("approve",      description="Type of action to take")
    field_name:    Optional[str]  = Field(None,           description="Field name to reveal (adaptive_inquiry only)")
    decision:      Optional[str]  = Field(None,           description="approve or deny")
    tier:          Optional[str]  = Field(None,           description="low_risk | medium_risk | high_risk")
    credit_limit:  Optional[float]= Field(None,           description="Suggested credit limit in INR")

# ── Observation ───────────────────────────────────────────────────────────────
class CreditObservation(Observation):
    """Everything the agent can see after each step."""
    applicant_id:      str              = Field("",                  description="Unique applicant identifier")
    revealed_fields:   Dict[str, float] = Field(default_factory=dict,description="Currently visible feature values")
    hidden_fields:     List[str]        = Field(default_factory=list, description="Fields the agent can request")
    task_name:         str              = Field("",                  description="Active task name")
    step_reward:       float            = Field(0.0,                 description="Reward for this step")
    cumulative_reward: float            = Field(0.0,                 description="Total reward so far")
    done:              bool             = Field(False,               description="Whether the episode has ended")
    message:           str              = Field("",                  description="Human-readable environment feedback")
    episode_score:     float            = Field(0.0,                 description="Final grader score (set when done=True)")
    
class CreditState:
    def __init__(
        self,
        episode_id=None,
        task_name="binary_decision",
        step_count=0,
        cumulative_reward=0.0,
        fields_requested=None,
        ground_truth_tier=None,
        ground_truth_decision=None,
        ground_truth_prob=None,
        done=False,
    ):
        self.episode_id = episode_id
        self.task_name = task_name
        self.step_count = step_count
        self.cumulative_reward = cumulative_reward
        self.fields_requested = fields_requested or []
        self.ground_truth_tier = ground_truth_tier
        self.ground_truth_decision = ground_truth_decision
        self.ground_truth_prob = ground_truth_prob
        self.done = done