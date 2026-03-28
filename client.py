# client.py
"""
WebSocket-based client for the CredLess Credit Analyst environment.
Uses openenv.core.env_client.EnvClient (persistent WS connection).

Usage (async):
    from client import CreditEnv
    from models import CreditAction

    async with CreditEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_name="adaptive_inquiry")
        print(result.observation.revealed_fields)
        result = await env.step(CreditAction(
            action_type="request_field", field_name="account_age"
        ))
        result = await env.step(CreditAction(
            action_type="approve", decision="approve"
        ))
        print(result.observation.episode_score)

Usage (sync):
    with CreditEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_name="binary_decision")
        result = env.step(CreditAction(action_type="approve", decision="approve"))
"""
from __future__ import annotations
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from models import CreditAction, CreditObservation


class CreditEnv(EnvClient[CreditAction, CreditObservation, State]):
    """
    Client for the CredLess Credit Analyst OpenEnv environment.
    Connects via WebSocket to the running server.
    """

    def _step_payload(self, action: CreditAction) -> dict:
        """Serialise action to the JSON payload the server expects."""
        payload = {"action_type": action.action_type}
        if action.field_name  is not None: payload["field_name"]   = action.field_name
        if action.decision    is not None: payload["decision"]     = action.decision
        if action.tier        is not None: payload["tier"]         = action.tier
        if action.credit_limit is not None: payload["credit_limit"] = action.credit_limit
        return payload

    def _parse_result(self, payload: dict) -> StepResult[CreditObservation]:
        """Parse server JSON response into typed StepResult."""
        obs_data = payload.get("observation", payload)
        obs = CreditObservation(
            applicant_id      = obs_data.get("applicant_id",      ""),
            revealed_fields   = obs_data.get("revealed_fields",   {}),
            hidden_fields     = obs_data.get("hidden_fields",     []),
            task_name         = obs_data.get("task_name",         ""),
            step_reward       = obs_data.get("step_reward",       0.0),
            cumulative_reward = obs_data.get("cumulative_reward", 0.0),
            done              = obs_data.get("done",              False),
            message           = obs_data.get("message",          ""),
            episode_score     = obs_data.get("episode_score",     0.0),
        )
        return StepResult(
            observation = obs,
            reward      = obs_data.get("step_reward", 0.0),
            done        = obs_data.get("done",        False),
        )

    def _parse_state(self, payload: dict) -> State:
        """Parse /state response into a State object."""
        return State(
            episode_id = payload.get("episode_id",  ""),
            step_count = payload.get("step_count",  0),
        )