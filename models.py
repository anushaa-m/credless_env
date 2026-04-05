# models.py
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class CreditAction(Action):
    action_type:   str            = Field("approve")
    field_name:    Optional[str]  = Field(None)
    decision:      Optional[str]  = Field(None)
    tier:          Optional[str]  = Field(None)
    credit_limit:  Optional[float]= Field(None)


class CreditObservation(Observation):
    applicant_id:      str              = Field("")
    revealed_fields:   Dict[str, float] = Field(default_factory=dict)
    hidden_fields:     List[str]        = Field(default_factory=list)
    task_name:         str              = Field("")
    step_reward:       float            = Field(0.0)
    cumulative_reward: float            = Field(0.0)
    done:              bool             = Field(False)
    message:           str              = Field("")
    episode_score:     float            = Field(0.0)


# ✅ NEW: Required by OpenEnv spec
class CreditReward(BaseModel):
    value:  float = Field(0.0,  description="Scalar reward for this step")
    reason: str   = Field("",   description="Human-readable reward explanation")


# ✅ FIXED: Now a Pydantic model so FastAPI can serialize it
class CreditState(BaseModel):
    episode_id:            str       = Field("")
    task_name:             str       = Field("binary_decision")
    step_count:            int       = Field(0)
    fields_requested:      List[str] = Field(default_factory=list)
    cumulative_reward:     float     = Field(0.0)
    ground_truth_tier:     str       = Field("")
    ground_truth_decision: str       = Field("")
    ground_truth_prob:     float     = Field(0.0)
    trajectory_length:     int       = Field(0)