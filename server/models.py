from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from models import FinVerseObservation


class FinVerseResponse(BaseModel):
    observation: FinVerseObservation
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    explanation: str = Field(default="")
    oracle_risk: float = Field(default=0.0)
    oracle_confidence: float = Field(default=0.0)
    top_factors: List[List[Any]] = Field(default_factory=list)
    session_id: str = Field(default="")
    episode_id: str = Field(default="")
    info: Dict[str, Any] = Field(default_factory=dict)
    portfolio_context: Dict[str, Any] = Field(default_factory=dict)
