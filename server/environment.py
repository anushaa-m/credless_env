from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import FinVerseObservation, FinVerseState
from .data_generator import FIELD_NAMES, generate_applicant
from .oracle import MARKET_SCENARIOS, CredLessOracle
from .tasks import TASK_DIFFICULTY, TASK_NAMES

MAX_STEPS = 5
VALID_ACTIONS = ["APPROVE", "REJECT"]
VALID_STEP_BONUS = 0.05
EFFICIENCY_PENALTY = 0.01
INVALID_ACTION_PENALTY = 0.5


class CreditAnalystEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.oracle = CredLessOracle()
        self._session_id = str(uuid.uuid4())
        self._episode_count = 0
        self._auditor_compliance_log: List[float] = []
        self._episode_id = ""
        self._task = "binary_decision"
        self._difficulty = "easy"
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._applicant: Dict[str, Any] = {}
        self._ground_truth: Dict[str, Any] = {}
        self._market_state: Dict[str, Any] = {}
        self._market_visible = False
        self._current_policy: Dict[str, Any] = {}
        self._revealed_fields: Dict[str, Dict[str, Any]] = {}
        self._done = False
        self._last_episode_score = 0.0
        self._oracle_risk = 0.5
        self._oracle_decision = "REJECT"

    def _build_policy(self, rng: random.Random) -> Dict[str, Any]:
        return {
            "valid_actions": list(VALID_ACTIONS),
            "step_contract": "Return only APPROVE or REJECT.",
            "dense_reward": VALID_STEP_BONUS,
            "efficiency_penalty": EFFICIENCY_PENALTY,
        }

    def _pick_market(self, rng: random.Random) -> Dict[str, Any]:
        name = rng.choice(list(MARKET_SCENARIOS.keys()))
        config = MARKET_SCENARIOS[name]
        return {
            "name": name,
            "base_rate": round(9.5 + (float(config["risk_multiplier"]) - 1.0) * 20.0, 2),
            "default_risk_index": round(float(config["risk_multiplier"]), 3),
            "sector_outlook": str(config["summary"]),
            "threshold_delta": round(float(config["threshold_delta"]), 3),
        }

    def _applicant_payload(self) -> Dict[str, Any]:
        return {
            "applicant_id": self._applicant.get("applicant_id", ""),
            "profile": dict(self._revealed_fields),
            "missing_fields": [],
            "declared_quality": self._applicant.get("data_quality", "observed_with_noise"),
            "source": self._applicant.get("source", "dataset_sample"),
        }

    def _build_observation(
        self,
        step_reward: float = 0.0,
        done: bool = False,
        episode_score: float = 0.0,
        message: str = "",
    ) -> FinVerseObservation:
        market_state = None
        if self._market_visible:
            market_state = {
                "base_rate": self._market_state["base_rate"],
                "default_risk_index": self._market_state["default_risk_index"],
                "sector_outlook": self._market_state["sector_outlook"],
                "name": self._market_state["name"],
            }

        return FinVerseObservation(
            applicant=self._applicant_payload(),
            conversation_history=[],
            market_visible=self._market_visible,
            market_state=market_state,
            current_policy=dict(self._current_policy),
            compliance_history=list(self._auditor_compliance_log[-3:]),
            step=self._steps_taken,
            max_steps=MAX_STEPS,
            fraud_flags_raised=[],
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            message=message,
            episode_score=round(episode_score, 4),
            task_name=self._task,
        )

    def _reveal_field(self, field: str, source: str) -> None:
        self._revealed_fields[field] = {
            "value": round(float(self._applicant["presented_features"][field]), 6),
            "confidence": round(float(self._applicant["field_confidence"][field]), 3),
            "source": source,
        }

    def reset(self, task_name: str = "binary_decision", seed: Optional[int] = None) -> FinVerseObservation:
        if task_name not in TASK_NAMES:
            task_name = "binary_decision"

        rng = random.Random(seed)
        self._task = task_name
        self._difficulty = TASK_DIFFICULTY.get(task_name, "easy")
        self._episode_count += 1
        self._episode_id = str(uuid.uuid4())
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_episode_score = 0.0
        self._applicant = generate_applicant(seed=seed, difficulty=self._difficulty)
        self._market_state = self._pick_market(rng)
        self._ground_truth = self.oracle.predict(self._applicant["features"], market_condition=self._market_state["name"])
        self._oracle_risk = self.oracle.predict_risk(
            self._applicant["features"],
            market_condition=self._market_state["name"],
        )
        self._oracle_decision = "REJECT" if self._oracle_risk >= 0.50 else "APPROVE"
        self._market_visible = True
        self._current_policy = self._build_policy(rng)
        self._revealed_fields = {}

        for field in FIELD_NAMES:
            self._reveal_field(field, source="initial")

        return self._build_observation(
            message=(
                "Environment ready. Submit exactly one uppercase action: APPROVE or REJECT."
            )
        )

    def _invalid_observation(self, message: str, reward: float) -> FinVerseObservation:
        self._cumulative_reward += reward
        return self._build_observation(step_reward=reward, message=message)

    def _ensure_active(self) -> Optional[Dict[str, object]]:
        if not self._episode_id:
            return {
                "reward": float(-(INVALID_ACTION_PENALTY + EFFICIENCY_PENALTY)),
                "done": True,
                "info": {
                    "explanation": "No active episode. Call reset() before step().",
                    "oracle_score": 0.0,
                    "oracle_risk": 0.0,
                    "oracle_decision": "",
                    "penalties_applied": {
                        "invalid_action": INVALID_ACTION_PENALTY,
                        "efficiency": EFFICIENCY_PENALTY,
                    },
                },
            }
        if self._done:
            return {
                "reward": 0.0,
                "done": True,
                "info": {
                    "explanation": "Episode already completed. Call reset() to start a new episode.",
                    "oracle_score": round(float(self._last_episode_score), 4),
                    "oracle_risk": round(float(self._oracle_risk), 4),
                    "oracle_decision": self._oracle_decision,
                    "penalties_applied": {"invalid_action": 0.0, "efficiency": 0.0},
                },
            }
        return None

    def _normalize_action(self, action: str) -> str:
        normalized = str(action or "").strip()
        assert normalized in VALID_ACTIONS
        return normalized

    def _oracle_explanation(self, action: str, oracle_score: float, penalties: Dict[str, float]) -> str:
        return (
            f"Action={action}. Oracle default risk={self._oracle_risk:.4f}, "
            f"oracle_decision={self._oracle_decision}, oracle_score={oracle_score:.4f}. "
            f"Penalties applied: invalid_action={penalties['invalid_action']:.2f}, "
            f"efficiency={penalties['efficiency']:.2f}. "
            "Terminal reward uses oracle_score by contract."
        )

    def _final_info(self, action: str, oracle_score: float, penalties: Dict[str, float]) -> Dict[str, object]:
        return {
            "explanation": self._oracle_explanation(action, oracle_score, penalties),
            "oracle_score": round(float(oracle_score), 4),
            "oracle_risk": round(float(self._oracle_risk), 4),
            "oracle_decision": self._oracle_decision,
            "penalties_applied": penalties,
            "configured_valid_step_bonus": VALID_STEP_BONUS,
            "reward_rule": "final reward = oracle_score when done=True",
        }

    def step(self, action: str) -> Dict[str, object]:
        guard = self._ensure_active()
        if guard is not None:
            return guard

        self._steps_taken += 1
        penalties = {"invalid_action": 0.0, "efficiency": 0.0}

        try:
            normalized_action = self._normalize_action(action)
        except AssertionError:
            penalties["efficiency"] = EFFICIENCY_PENALTY
            penalties["invalid_action"] = INVALID_ACTION_PENALTY
            reward = -(INVALID_ACTION_PENALTY + EFFICIENCY_PENALTY)
            done = self._steps_taken >= MAX_STEPS
            if done:
                self._done = True
                self._last_episode_score = 0.0
            return {
                "reward": round(float(reward), 4),
                "done": done,
                "info": self._final_info("INVALID", 0.0, penalties),
            }

        oracle_score = (1.0 - float(self._oracle_risk)) if normalized_action == "APPROVE" else float(self._oracle_risk)
        reward = float(np.clip(oracle_score, 0.0, 1.0))
        self._cumulative_reward += reward
        self._done = True
        self._last_episode_score = reward
        self._auditor_compliance_log.append(reward)
        return {
            "reward": round(reward, 4),
            "done": True,
            "info": self._final_info(normalized_action, reward, penalties),
        }

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
            auditor_compliance_log=list(self._auditor_compliance_log),
            episode_count=self._episode_count,
        )
