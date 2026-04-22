from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from pipeline.main_pipeline import CreditDecisionPipeline
from rl.reward_logger import RewardLogger


@dataclass
class Transition:
    episode_id: str
    step_index: int
    observation: dict[str, Any]
    action: str
    reward: float
    done: bool
    logprob: float
    info: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class Trajectory:
    episode_id: str
    transitions: list[Transition]
    total_reward: float
    done: bool
    user_data: dict[str, Any]
    summary: dict[str, Any]


class RolloutCollector:
    def __init__(
        self,
        pipeline: CreditDecisionPipeline | None = None,
        *,
        reward_logger: RewardLogger | None = None,
    ) -> None:
        self.pipeline = pipeline or CreditDecisionPipeline()
        self.reward_logger = reward_logger

    def collect(
        self,
        users: Iterable[Mapping[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for index, user_data in enumerate(users):
            if limit is not None and index >= limit:
                break
            trajectory = self.collect_one(user_data)
            trajectories.append(trajectory)
        return trajectories

    def collect_one(self, user_data: Mapping[str, Any]) -> Trajectory:
        episode_id = str(uuid.uuid4())
        result = self.pipeline.run(dict(user_data))

        transition = Transition(
            episode_id=episode_id,
            step_index=0,
            observation={
                "features": dict(result["user_data"]),
                "risk_score": float(result["risk_score"]),
                "shap_info": list(result["shap_info"]),
            },
            action=str(result["decision"]),
            reward=float(result["reward"]),
            done=bool(result["done"]),
            logprob=float(result["policy_output"]["logprob"]),
            info=dict(result["info"]),
            metadata={
                "prompt": result["policy_output"]["prompt"],
                "raw_text": result["policy_output"]["raw_text"],
                "approve_probability": float(result["policy_output"]["approve_probability"]),
            },
        )
        summary = {
            "episode_id": episode_id,
            "total_reward": float(result["reward"]),
            "done": bool(result["done"]),
            "decision": str(result["decision"]),
            "risk_score": float(result["risk_score"]),
            "oracle_score": float(result["info"].get("oracle_score", 0.0)),
        }
        trajectory = Trajectory(
            episode_id=episode_id,
            transitions=[transition],
            total_reward=float(result["reward"]),
            done=bool(result["done"]),
            user_data=dict(result["user_data"]),
            summary=summary,
        )

        if self.reward_logger is not None:
            self.reward_logger.log_step(asdict(transition))
            self.reward_logger.log_episode(summary)

        return trajectory
