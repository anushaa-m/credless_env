# client.py
"""
WebSocket-based client for the FinVerse credit investigation environment.
"""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State

from models import FinVerseAction, FinVerseObservation


class CreditEnv(EnvClient[Any, FinVerseObservation, State]):
    def _step_payload(self, action: Any) -> Any:
        if isinstance(action, FinVerseAction):
            return action.model_dump()
        if isinstance(action, dict):
            return action
        raw = str(action).strip()
        if raw.upper() in {"APPROVE", "REJECT"}:
            return raw.upper()
        return raw.lower()

    def _parse_result(self, payload: dict) -> StepResult[FinVerseObservation]:
        obs_data = payload.get("observation", {})
        obs = FinVerseObservation.model_validate(obs_data)
        return StepResult(
            observation=obs,
            reward=float(payload.get("reward", obs.step_reward)),
            done=bool(payload.get("done", obs.done)),
        )

    def _parse_state(self, payload: dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("steps_taken", 0),
        )
