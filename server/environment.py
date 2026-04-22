from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import FinVerseObservation, FinVerseState
from .data_generator import generate_applicant
from .oracle import MARKET_SCENARIOS, CredLessOracle
from .tasks import TASK_DIFFICULTY, TASK_NAMES


INVALID_ACTION_PENALTY = -0.5
VALID_ACTIONS = {"APPROVE", "REJECT"}


class CreditAnalystEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.oracle = CredLessOracle()
        self._session_id = str(uuid.uuid4())
        self._episode_count = 0
        self._episode_id = ""
        self._task = "binary_decision"
        self._difficulty = "easy"
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._applicant: Dict[str, Any] = {}
        self._ground_truth: Dict[str, Any] = {}
        self._market_state: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}
        self._done = False
        self.trajectory: List[Dict[str, Any]] = []

    def last_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def reset(self, task_name: str = "binary_decision", seed: Optional[int] = None) -> FinVerseObservation:
        if task_name not in TASK_NAMES:
            task_name = "binary_decision"

        self._task = task_name
        self._difficulty = TASK_DIFFICULTY.get(task_name, "easy")
        self._episode_count += 1
        self._episode_id = str(uuid.uuid4())
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._done = False
        self.trajectory = []
        self._applicant = generate_applicant(seed=seed, difficulty=self._difficulty)
        self._market_state = MARKET_SCENARIOS.get("Stable Credit", {})
        self._ground_truth = self.oracle.predict(self._applicant["features"], market_condition="Stable Credit")

        default_prob = float(self._ground_truth.get("default_prob", 0.5))
        oracle_decision = "APPROVE" if str(self._ground_truth.get("decision", "")).lower() == "approve" else "REJECT"
        oracle_confidence = 1.0 - default_prob if oracle_decision == "APPROVE" else default_prob
        self._last_info = {
            "explanation": "Episode reset.",
            "oracle_score": 0.0,
            "oracle_decision": oracle_decision,
            "oracle_confidence": float(oracle_confidence),
        }

        return FinVerseObservation(
            applicant={
                "applicant_id": self._applicant.get("applicant_id", ""),
                "profile": dict(self._applicant.get("features", {})),
                "missing_fields": [],
                "declared_quality": self._applicant.get("data_quality", "observed_with_noise"),
                "source": self._applicant.get("source", "dataset_sample"),
            },
            conversation_history=[],
            market_visible=False,
            market_state=None,
            current_policy={"allowed_actions": ["APPROVE", "REJECT"]},
            compliance_history=[],
            step=0,
            max_steps=1,
            fraud_flags_raised=[],
            step_reward=0.0,
            cumulative_reward=0.0,
            done=False,
            message="Single-step binary decision environment. Respond with APPROVE or REJECT.",
            episode_score=0.0,
            task_name=self._task,
        )

    def step(self, action: str) -> Dict[str, Any]:
        if not self._episode_id:
            result = {
                "reward": float(INVALID_ACTION_PENALTY),
                "done": True,
                "info": {
                    "explanation": "Environment not reset.",
                    "oracle_score": 0.0,
                    "oracle_decision": "REJECT",
                    "oracle_confidence": 0.0,
                    "error": "reset required",
                },
            }
            self._last_info = dict(result["info"])
            return result

        normalized_action = str(action or "").strip().upper()
        self._steps_taken = 1
        self._done = True

        oracle_raw_decision = str(self._ground_truth.get("decision", "")).lower()
        oracle_decision = "APPROVE" if oracle_raw_decision == "approve" else "REJECT"
        default_prob = float(self._ground_truth.get("default_prob", 0.5))
        oracle_confidence = 1.0 - default_prob if oracle_decision == "APPROVE" else default_prob
        explanation_payload = self.oracle.explain_decision(
            self._applicant["features"],
            market_condition="Stable Credit",
        )
        explanation = str(explanation_payload.get("explanation", ""))

        if normalized_action not in VALID_ACTIONS:
            result = {
                "reward": float(INVALID_ACTION_PENALTY),
                "done": True,
                "info": {
                    "explanation": explanation,
                    "oracle_score": float(INVALID_ACTION_PENALTY),
                    "oracle_decision": oracle_decision,
                    "oracle_confidence": float(oracle_confidence),
                    "error": "invalid action",
                },
            }
            self._cumulative_reward = float(INVALID_ACTION_PENALTY)
            self._last_info = dict(result["info"])
            self.trajectory.append({"step": 1, "action": normalized_action, "reward": float(INVALID_ACTION_PENALTY)})
            return result

        reward = float(oracle_confidence if normalized_action == oracle_decision else 1.0 - oracle_confidence)
        result = {
            "reward": reward,
            "done": True,
            "info": {
                "explanation": explanation,
                "oracle_score": reward,
                "oracle_decision": oracle_decision,
                "oracle_confidence": float(oracle_confidence),
            },
        }
        self._cumulative_reward = reward
        self._last_info = dict(result["info"])
        self.trajectory.append({"step": 1, "action": normalized_action, "reward": reward})
        return result

    def state(self) -> FinVerseState:
        return FinVerseState(
            session_id=self._session_id,
            episode_id=self._episode_id,
            task_difficulty=self._difficulty,
            applicant_ground_truth=dict(self._applicant.get("features", {})),
            applicant_is_fraudulent=bool(self._applicant.get("is_adversarial", False)),
            market_state=dict(self._market_state),
            conversation=[],
            fraud_flags=[],
            steps_taken=self._steps_taken,
            auditor_compliance_log=[self._cumulative_reward] if self._episode_id else [],
            episode_count=self._episode_count,
        )
