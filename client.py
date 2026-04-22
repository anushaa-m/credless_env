from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State

    OPENENV_AVAILABLE = True

except Exception:
    StepResult = dict
    State = dict

    class EnvClient:
        pass

    OPENENV_AVAILABLE = False


# 👇 THIS LINE FIXES YOUR ERROR
if OPENENV_AVAILABLE:
    BaseEnv = EnvClient[str, "FinVerseObservation", State]
else:
    BaseEnv = EnvClient


class CreditEnv(BaseEnv):
    def _step_payload(self, action: str) -> str:
        return str(action).strip().upper()

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
