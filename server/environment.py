from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import FinVerseAction, FinVerseObservation, FinVerseState
from pipeline.main_pipeline import FrozenRiskPredictor
from .data_generator import FIELD_NAMES, generate_applicant
from .oracle import MARKET_SCENARIOS, CredLessOracle
from .tasks import TASK_DIFFICULTY, TASK_NAMES

MAX_STEPS = 8
VALID_FIELD_REQUEST_REWARD = 0.05
STEP_PENALTY = 0.01
INVALID_ACTION_PENALTY = 0.5
POLICY_REQUIRED_FIELDS = {
    "easy": ["payment_reliability", "debt_burden_score"],
    "medium": ["payment_reliability", "debt_burden_score", "overdraft_risk"],
    "hard": ["payment_reliability", "debt_burden_score", "overdraft_risk", "total_delinquency_score"],
}


class CreditAnalystEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.oracle = CredLessOracle()
        self.risk_predictor = FrozenRiskPredictor()
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
        self._conversation: List[Dict[str, Any]] = []
        self._fraud_flags: List[str] = []
        self._requested_fields: List[str] = []
        self._revealed_fields: Dict[str, Dict[str, Any]] = {}
        self._done = False
        self._last_episode_score = 0.0
        self._last_info: Dict[str, Any] = {}
        self._current_observation: Dict[str, Any] = {}
        self.trajectory: List[Any] = []

    def _build_policy(self, rng: random.Random) -> Dict[str, Any]:
        max_dti = {
            "easy": 0.60,
            "medium": 0.55,
            "hard": 0.50,
        }[self._difficulty]
        max_dti += rng.uniform(-0.03, 0.03)
        return {
            "max_dti": round(max(0.35, min(0.70, max_dti)), 3),
            "required_fields": list(POLICY_REQUIRED_FIELDS[self._difficulty]),
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
            "missing_fields": [field for field in FIELD_NAMES if field not in self._revealed_fields],
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
            conversation_history=list(self._conversation),
            market_visible=self._market_visible,
            market_state=market_state,
            current_policy=dict(self._current_policy),
            compliance_history=list(self._auditor_compliance_log[-3:]),
            step=self._steps_taken,
            max_steps=MAX_STEPS,
            fraud_flags_raised=list(self._fraud_flags),
            step_reward=round(step_reward, 4),
            cumulative_reward=round(self._cumulative_reward, 4),
            done=done,
            message=message,
            episode_score=round(episode_score, 4),
            task_name=self._task,
        )

    def _append_conversation(self, role: str, content: str) -> None:
        self._conversation.append(
            {
                "role": role,
                "content": content,
                "step": self._steps_taken,
            }
        )

    def _reveal_field(self, field: str, source: str) -> None:
        self._revealed_fields[field] = {
            "value": round(float(self._applicant["presented_features"][field]), 6),
            "confidence": round(float(self._applicant["field_confidence"][field]), 3),
            "source": source,
        }

    def _set_last_info(self, explanation: str, oracle_score: float, penalties: Dict[str, float] | None = None) -> None:
        self._last_info = {
            "task_name": self._task,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "episode_score": round(self._last_episode_score, 4),
            "market_visible": self._market_visible,
            "fraud_flags_raised": list(self._fraud_flags),
            "explanation": explanation,
            "oracle_score": round(float(oracle_score), 4),
            "penalties_applied": penalties or {},
        }

    def last_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    def _current_features(self) -> Dict[str, float]:
        return {
            field: round(float(self._applicant["presented_features"][field]), 6)
            for field in FIELD_NAMES
        }

    def _build_binary_observation(self) -> Dict[str, Any]:
        features = self._current_features()
        shap_items = self.risk_predictor.explain(features)
        observation = {
            "features": features,
            "risk_score": round(float(self.risk_predictor.predict(features)), 6),
            "shap_info": {"top_features": shap_items},
        }
        self._current_observation = observation
        return observation

    def _error_result(self, message: str, reward: float = -1.0, oracle_score: float = 0.0) -> Dict[str, Any]:
        return {
            "reward": float(reward),
            "done": True,
            "info": {
                "explanation": message,
                "oracle_score": round(float(oracle_score), 4),
                "oracle_decision": "REJECT",
                "oracle_confidence": 0.0,
            },
        }

    def _compute_exploration_bonus(self) -> float:
        top_features = self._current_observation.get("shap_info", {}).get("top_features", [])
        unique_features = {
            str(item.get("feature", "")).strip()
            for item in top_features
            if str(item.get("feature", "")).strip()
        }
        if not unique_features:
            return 0.0
        return min(0.05, 0.015 * len(unique_features))

    def reset(self, task_name: str = "binary_decision", seed: Optional[int] = None) -> Dict[str, Any]:
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
        self._ground_truth = self.oracle.predict(
            self._applicant["features"],
            market_condition=self._market_state["name"],
        )
        self._market_visible = False
        self._current_policy = self._build_policy(rng)
        self._conversation = []
        self._fraud_flags = []
        self._requested_fields = []
        self._revealed_fields = {}
        self.trajectory = []

        for field in self._applicant.get("visible_fields", []):
            self._reveal_field(field, source="initial")

        self._append_conversation(
            "system",
            (
                f"Episode {self._episode_id} started for task '{self._task}' with difficulty "
                f"'{self._difficulty}'. Investigate the applicant within {MAX_STEPS} steps."
            ),
        )
        self._append_conversation(
            "applicant",
            (
                f"Submitting application {self._applicant['applicant_id']} with partially observed "
                f"profile data. Declared quality: {self._applicant.get('data_quality', 'observed_with_noise')}."
            ),
        )
        self._set_last_info("Episode reset.", 0.0, {})
        return self._build_binary_observation()

    def _invalid(self, message: str, penalty: float = INVALID_ACTION_PENALTY) -> FinVerseObservation:
        reward = -(penalty + STEP_PENALTY)
        self._cumulative_reward += reward
        self._set_last_info(
            message,
            0.0,
            {"invalid_action": penalty, "efficiency": STEP_PENALTY},
        )
        return self._build_observation(step_reward=reward, message=message)

    def _ensure_active(self) -> Optional[FinVerseObservation]:
        if not self._episode_id:
            self._set_last_info("ERROR: call /reset before /step.", 0.0, {"invalid_action": INVALID_ACTION_PENALTY})
            return FinVerseObservation(
                step_reward=-(INVALID_ACTION_PENALTY + STEP_PENALTY),
                done=True,
                message="ERROR: call /reset before /step.",
                episode_score=0.0,
                task_name=self._task,
            )
        if self._done:
            self._set_last_info(
                "Episode already completed. Call /reset to start a new episode.",
                self._last_episode_score,
                {},
            )
            return self._build_observation(
                step_reward=0.0,
                done=True,
                episode_score=self._last_episode_score,
                message="Episode already completed. Call /reset to start a new episode.",
            )
        return None

    def _coerce_action(self, action: str | FinVerseAction) -> FinVerseAction:
        if isinstance(action, FinVerseAction):
            return action
        if isinstance(action, dict):
            action_type = str(action.get("action_type", "")).strip()
            params = action.get("params", {}) or {}
            reasoning = str(action.get("reasoning", "") or "")
            normalized = action_type.lower()
            if normalized == "request_more_info":
                normalized = "request_info"
            return FinVerseAction(action_type=normalized, params=params, reasoning=reasoning)
        raw = str(action or "").strip()
        if raw == "APPROVE":
            return FinVerseAction(action_type="approve", params={}, reasoning="")
        if raw == "REJECT":
            return FinVerseAction(action_type="deny", params={}, reasoning="")
        raise ValueError(f"Unsupported action '{raw}'.")

    def _oracle_score_for_decision(self, decision: str) -> float:
        default_prob = float(self._ground_truth.get("default_prob", 0.5))
        if decision == "approve":
            return round(max(0.0, min(1.0, 1.0 - default_prob)), 4)
        if decision == "deny":
            return round(max(0.0, min(1.0, default_prob)), 4)
        return 0.0

    def _handle_request_info(self, action: FinVerseAction) -> FinVerseObservation:
        field = str(action.params.get("field", "")).strip()
        if not field:
            return self._invalid("request_info requires params.field.")
        if field not in FIELD_NAMES:
            return self._invalid(f"Unknown applicant field '{field}'.")
        if field in self._revealed_fields:
            return self._invalid(f"Field '{field}' is already visible.", penalty=0.10)

        self._requested_fields.append(field)
        self._append_conversation("assistant", f"Please provide the field '{field}'.")
        self._reveal_field(field, source="requested")
        profile_entry = self._revealed_fields[field]
        self._append_conversation(
            "applicant",
            (
                f"{field} provided as {profile_entry['value']:.6f} "
                f"with self-reported confidence {profile_entry['confidence']:.2f}."
            ),
        )

        reward = VALID_FIELD_REQUEST_REWARD - STEP_PENALTY
        self._cumulative_reward += reward
        self._set_last_info(
            f"Revealed '{field}' from applicant response.",
            0.0,
            {"valid_field_request": VALID_FIELD_REQUEST_REWARD, "efficiency": STEP_PENALTY},
        )
        return self._build_observation(
            step_reward=reward,
            message=f"Revealed '{field}' from applicant response.",
        )

    def _handle_query_market(self) -> FinVerseObservation:
        if self._market_visible:
            return self._invalid("Market state is already visible.", penalty=0.10)

        self._market_visible = True
        self._append_conversation("assistant", "Requesting current lending market conditions.")
        self._append_conversation(
            "system",
            (
                f"Market revealed: {self._market_state['name']} with base rate "
                f"{self._market_state['base_rate']} and outlook '{self._market_state['sector_outlook']}'."
            ),
        )
        reward = -STEP_PENALTY
        self._cumulative_reward += reward
        self._set_last_info(
            "Market conditions revealed.",
            0.0,
            {"efficiency": STEP_PENALTY},
        )
        return self._build_observation(
            step_reward=reward,
            message="Market conditions revealed at the cost of one investigation step.",
        )

    def _handle_flag_fraud(self, action: FinVerseAction) -> FinVerseObservation:
        reason = str(action.params.get("reason", "") or action.reasoning).strip()
        if not reason:
            return self._invalid("flag_fraud requires params.reason or reasoning.")
        if reason in self._fraud_flags:
            return self._invalid("This fraud flag has already been raised.", penalty=0.10)

        self._fraud_flags.append(reason)
        self._append_conversation("assistant", f"Fraud flag raised: {reason}")
        reward = -STEP_PENALTY
        self._cumulative_reward += reward
        self._set_last_info(
            "Fraud flag recorded for downstream review.",
            0.0,
            {"efficiency": STEP_PENALTY},
        )
        return self._build_observation(
            step_reward=reward,
            message="Fraud flag recorded for downstream review.",
        )

    def _handle_terminal(self, action: FinVerseAction) -> FinVerseObservation:
        oracle_score = self._oracle_score_for_decision(action.action_type)
        explanation = self.oracle.explain_decision(
            self._applicant["features"],
            market_condition=self._market_state["name"],
        )

        # Base dense reward from oracle
        reward = oracle_score if action.action_type in {"approve", "deny"} else 0.0

# Convert to centered range (-1 to +1)
        reward = (reward * 2) - 1
        reward=reward*2

# Get oracle decision
        oracle_decision = self._ground_truth.get("decision")

# Add correctness shaping
        if action.action_type == oracle_decision:
            reward += 0.5
        else:
            reward -= 0.5

# Add confidence shaping
        confidence = float(self._ground_truth.get("confidence", 0.5))
        if action.action_type == oracle_decision:
                reward += 0.3 * confidence

# Efficiency penalty
        reward -= 0.01 * self._steps_taken

# Clamp reward for stability    
        reward = max(-1.5, min(1.5, reward))
        self._cumulative_reward += reward
        self._done = True
        self._last_episode_score = reward
        self._auditor_compliance_log.append(reward)
        self._append_conversation(
            "assistant",
            f"Final action: {action.action_type}. Reasoning: {action.reasoning.strip()}",
        )
        self._append_conversation(
            "system",
            (
                f"Episode resolved with oracle_decision={self._ground_truth['decision']} "
                f"and oracle_score={oracle_score:.4f}."
            ),
        )
        self._set_last_info(
            explanation["explanation"],
            oracle_score,
            {"final_reward": oracle_score},
        )
        return self._build_observation(
            step_reward=reward,
            done=True,
            episode_score=reward,
            message=explanation["explanation"],
        )

    def step(self, action: str | FinVerseAction) -> Dict[str, Any]:
        if not self._episode_id:
            return self._error_result("ERROR: call reset() before step().")
        if self._done:
            return self._error_result("Episode already completed. Call reset() to start a new episode.", reward=0.0)

        self._steps_taken += 1
        raw_action = str(action or "").strip()
        if raw_action not in {"APPROVE", "REJECT"}:
            self._done = True
            self.trajectory.append({"step": self._steps_taken, "action": raw_action})
            return self._error_result("Action must be exactly APPROVE or REJECT.", reward=-1.25)

        self.trajectory.append({"step": self._steps_taken, "action": raw_action})
        chosen_action = "approve" if raw_action == "APPROVE" else "deny"
        oracle_decision = "APPROVE" if str(self._ground_truth.get("decision", "deny")).lower() == "approve" else "REJECT"
        default_prob = float(self._ground_truth.get("default_prob", 0.5))
        oracle_confidence = (1.0 - default_prob) if oracle_decision == "APPROVE" else default_prob
        oracle_confidence = float(max(0.0, min(1.0, oracle_confidence)))
        oracle_score = (1.0 - default_prob) if raw_action == "APPROVE" else default_prob
        oracle_score = float(max(0.0, min(1.0, oracle_score)))
        matches_oracle = raw_action == oracle_decision

        dense_reward = (2.0 * oracle_score) - 1.0
        correctness_bonus = 0.20 if matches_oracle else -0.20
        confidence_bonus = 0.20 * oracle_confidence if matches_oracle else -0.20 * oracle_confidence

        if raw_action == "APPROVE" and oracle_decision == "APPROVE":
            sparse_final = 0.45
        elif raw_action == "APPROVE" and oracle_decision == "REJECT":
            sparse_final = -0.65
        elif raw_action == "REJECT" and oracle_decision == "APPROVE":
            sparse_final = -0.35
        else:
            sparse_final = 0.35

        efficiency_penalty = 0.01 * self._steps_taken
        exploration_bonus = self._compute_exploration_bonus()
        reward = dense_reward + correctness_bonus + confidence_bonus + sparse_final + exploration_bonus - efficiency_penalty
        reward = float(max(-1.5, min(1.5, reward)))

        explanation = self.oracle.explain_decision(
            self._applicant["features"],
            market_condition=self._market_state["name"],
        )["explanation"]
        self._cumulative_reward += reward
        self._done = True
        self._last_episode_score = reward
        self._auditor_compliance_log.append(reward)
        self._last_info = {
            "explanation": explanation,
            "oracle_score": round(oracle_score, 4),
            "oracle_decision": oracle_decision,
            "oracle_confidence": round(oracle_confidence, 4),
        }
        return {
            "reward": reward,
            "done": True,
            "info": dict(self._last_info),
        }

    def state(self) -> FinVerseState:
        return FinVerseState(
            session_id=self._session_id,
            episode_id=self._episode_id,
            task_difficulty=self._difficulty,
            applicant_ground_truth=dict(self._applicant.get("features", {})),
            applicant_is_fraudulent=bool(self._applicant.get("is_adversarial", False)),
            market_state=dict(self._market_state),
            conversation=list(self._conversation),
            fraud_flags=list(self._fraud_flags),
            steps_taken=self._steps_taken,
            auditor_compliance_log=list(self._auditor_compliance_log),
            episode_count=self._episode_count,
        )
