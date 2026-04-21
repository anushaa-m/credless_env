from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


TerminalAction = Literal["approve", "deny", "escalate"]
FinVerseActionType = Literal[
    "request_info",
    "query_market",
    "flag_fraud",
    "approve",
    "deny",
    "escalate",
]


class FinVerseAction(Action):
    action_type: FinVerseActionType
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str = Field(default="")


class FinVerseObservation(Observation):
    applicant: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    market_visible: bool = Field(default=False)
    market_state: Optional[Dict[str, Any]] = Field(default=None)
    current_policy: Dict[str, Any] = Field(default_factory=dict)
    compliance_history: List[float] = Field(default_factory=list)
    step: int = Field(default=0)
    max_steps: int = Field(default=8)
    fraud_flags_raised: List[str] = Field(default_factory=list)
    step_reward: float = Field(default=0.0)
    cumulative_reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    message: str = Field(default="")
    episode_score: float = Field(default=0.0)
    task_name: str = Field(default="binary_decision")


class FinVerseReward(BaseModel):
    value: float = Field(default=0.0, description="Scalar reward for this step")
    reason: str = Field(default="", description="Human-readable reward explanation")


class FinVerseState(BaseModel):
    session_id: str = Field(default="")
    episode_id: str = Field(default="")
    task_difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    applicant_ground_truth: Dict[str, Any] = Field(default_factory=dict)
    applicant_is_fraudulent: bool = Field(default=False)
    market_state: Dict[str, Any] = Field(default_factory=dict)
    conversation: List[Dict[str, Any]] = Field(default_factory=list)
    fraud_flags: List[str] = Field(default_factory=list)
    steps_taken: int = Field(default=0)
    auditor_compliance_log: List[float] = Field(default_factory=list)
    episode_count: int = Field(default=0)


CreditAction = FinVerseAction
CreditObservation = FinVerseObservation
CreditReward = FinVerseReward
CreditState = FinVerseState
