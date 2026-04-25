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
VALID_MARKET_QUERY_REWARD = 0.05
VALID_FRAUD_FLAG_REWARD = 0.05
STEP_PENALTY = 0.01
INVALID_ACTION_PENALTY = 0.5
TIMEOUT_PENALTY = 1.0
DUPLICATE_ACTION_PENALTY = 0.10
NO_PROGRESS_LOOP_PENALTY = 0.15
FREE_EXTRA_FIELD_REQUESTS = 2
EXCESS_FIELD_REQUEST_PENALTY = 0.03
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
            action_history=list(self.trajectory),
            market_visible=self._market_visible,
            market_queried=self._market_visible,
            fraud_checked=bool(self._fraud_flags),
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

    def _set_last_info(
        self,
        explanation: str,
        oracle_score: float,
        penalties: Dict[str, float] | None = None,
        *,
        oracle_decision: str | None = None,
        oracle_confidence: float | None = None,
    ) -> None:
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
        if oracle_decision is not None:
            self._last_info["oracle_decision"] = str(oracle_decision).upper()
        if oracle_confidence is not None:
            self._last_info["oracle_confidence"] = round(float(oracle_confidence), 4)

    def last_info(self) -> Dict[str, Any]:
        return dict(self._last_info)

    @property
    def revealed_features(self) -> Dict[str, float]:
        return {
            field: round(float(payload.get("value", 0.0)), 6)
            for field, payload in self._revealed_fields.items()
        }

    def oracle_features(self) -> Dict[str, float]:
        return dict(self.revealed_features)

    def current_feature_snapshot(self) -> Dict[str, float]:
        return dict(self._current_features())

    def _current_features(self) -> Dict[str, float]:
        return {
            field: round(float(self._applicant["presented_features"][field]), 6)
            for field in FIELD_NAMES
        }

    def _refresh_current_observation(self) -> Dict[str, Any]:
        features = self._current_features()
        shap_items = self.risk_predictor.explain(features)
        observation = {
            "features": features,
            "risk_score": round(float(self.risk_predictor.predict(features)), 6),
            "shap_info": {"top_features": shap_items},
        }
        self._current_observation = observation
        return observation

    def _record_action(self, action: FinVerseAction) -> None:
        self.trajectory.append(
            {
                "step": self._steps_taken,
                "action_type": action.action_type,
                "params": dict(action.params),
                "reasoning": action.reasoning,
                "signature": self._action_signature(action),
                "progress_made": False,
            }
        )

    def _action_signature(self, action: FinVerseAction) -> str:
        if action.action_type == "request_info":
            return f"request_info:{str(action.params.get('field', '')).strip().lower()}"
        if action.action_type == "flag_fraud":
            reason = str(action.params.get("reason", "") or action.reasoning).strip().lower()
            return f"flag_fraud:{reason}"
        return action.action_type

    def _signature_repeat_penalty(self, signature: str) -> float:
        repeats = sum(1 for item in self.trajectory[:-1] if item.get("signature") == signature)
        if repeats <= 0:
            return 0.0
        return DUPLICATE_ACTION_PENALTY * repeats

    def _mark_last_action_progress(self, progress_made: bool) -> None:
        if self.trajectory:
            self.trajectory[-1]["progress_made"] = bool(progress_made)

    def _no_progress_repeat_penalty(self, action: FinVerseAction) -> float:
        if action.action_type in {"approve", "deny", "escalate"}:
            return 0.0
        no_progress_streak = 0
        for item in reversed(self.trajectory[:-1]):
            if item.get("action_type") != action.action_type:
                break
            if item.get("progress_made", False):
                break
            no_progress_streak += 1
        if no_progress_streak <= 0:
            return 0.0
        return NO_PROGRESS_LOOP_PENALTY * no_progress_streak

    def _apply_repeat_penalties(self, action: FinVerseAction) -> float:
        penalties_to_apply: Dict[str, float] = {}
        duplicate_penalty = self._signature_repeat_penalty(self._action_signature(action))
        if duplicate_penalty > 0.0:
            penalties_to_apply["duplicate_action"] = round(duplicate_penalty, 4)
        no_progress_penalty = self._no_progress_repeat_penalty(action)
        if no_progress_penalty > 0.0:
            penalties_to_apply["no_progress_loop"] = round(no_progress_penalty, 4)
        total_penalty = round(sum(penalties_to_apply.values()), 4)
        if total_penalty <= 0.0:
            return 0.0
        self._cumulative_reward -= total_penalty
        penalties = dict(self._last_info.get("penalties_applied", {}))
        penalties.update(penalties_to_apply)
        self._last_info["penalties_applied"] = penalties
        self._last_info["cumulative_reward"] = round(self._cumulative_reward, 4)
        return total_penalty

    def _record_reward_components(self, **components: float) -> Dict[str, float]:
        filtered = {name: round(float(value), 4) for name, value in components.items() if abs(float(value)) > 0.0}
        self._last_info["reward_components"] = filtered
        return filtered

    def _wrap_step_result(self, observation: FinVerseObservation) -> Dict[str, Any]:
        return {
            "observation": observation.model_dump(),
            "reward": round(float(observation.step_reward), 4),
            "done": bool(observation.done),
            "info": dict(self._last_info),
        }

    def _finalize_timeout(self, observation: FinVerseObservation) -> FinVerseObservation:
        reward = float(observation.step_reward) - TIMEOUT_PENALTY
        self._cumulative_reward -= TIMEOUT_PENALTY
        self._done = True
        self._last_episode_score = reward
        self._auditor_compliance_log.append(reward)
        penalties = dict(self._last_info.get("penalties_applied", {}))
        penalties["timeout"] = TIMEOUT_PENALTY
        self._set_last_info(
            "Maximum investigation steps reached before a terminal decision.",
            self._last_info.get("oracle_score", 0.0),
            penalties,
        )
        reward_components = dict(self._last_info.get("reward_components", {}))
        reward_components["timeout"] = round(reward_components.get("timeout", 0.0) - TIMEOUT_PENALTY, 4)
        self._last_info["reward_components"] = reward_components
        return self._build_observation(
            step_reward=reward,
            done=True,
            episode_score=reward,
            message="Maximum investigation steps reached before a terminal decision.",
        )

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
        self._refresh_current_observation()
        self._set_last_info("Episode reset.", 0.0, {})
        return self._build_observation(message="Episode reset. Investigate before making a terminal decision.").model_dump()

    def _invalid(self, message: str, penalty: float = INVALID_ACTION_PENALTY) -> FinVerseObservation:
        reward = -(penalty + STEP_PENALTY)
        self._cumulative_reward += reward
        self._set_last_info(
            message,
            0.0,
            {"invalid_action": penalty, "efficiency": STEP_PENALTY},
        )
        self._record_reward_components(
            invalid_action=-penalty,
            efficiency=-STEP_PENALTY,
            format_compliance=-penalty,
        )
        self._mark_last_action_progress(False)
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
        normalized = raw.lower()
        if raw == "APPROVE" or normalized == "approve":
            return FinVerseAction(action_type="approve", params={}, reasoning="")
        if raw == "REJECT" or normalized in {"reject", "deny"}:
            return FinVerseAction(action_type="deny", params={}, reasoning="")
        if normalized in {"query_market", "escalate"}:
            return FinVerseAction(action_type=normalized, params={}, reasoning="")
        if normalized == "flag_fraud":
            return FinVerseAction(action_type="flag_fraud", params={}, reasoning="")
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

        excess_requests = max(
            0,
            len(self._requested_fields) - len(self._current_policy.get("required_fields", [])) - FREE_EXTRA_FIELD_REQUESTS,
        )
        over_collection_penalty = EXCESS_FIELD_REQUEST_PENALTY * excess_requests
        reward = VALID_FIELD_REQUEST_REWARD - STEP_PENALTY - over_collection_penalty
        self._refresh_current_observation()
        self._cumulative_reward += reward
        penalties = {"valid_field_request": VALID_FIELD_REQUEST_REWARD, "efficiency": STEP_PENALTY}
        if over_collection_penalty > 0.0:
            penalties["excess_data_request"] = round(over_collection_penalty, 4)
        self._set_last_info(
            f"Revealed '{field}' from applicant response.",
            0.0,
            penalties,
        )
        self._record_reward_components(
            valid_field_request=VALID_FIELD_REQUEST_REWARD,
            efficiency=-STEP_PENALTY,
            excess_data_request=-over_collection_penalty,
        )
        self._mark_last_action_progress(True)
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
        reward = VALID_MARKET_QUERY_REWARD - STEP_PENALTY
        self._refresh_current_observation()
        self._cumulative_reward += reward
        self._set_last_info(
            "Market conditions revealed.",
            0.0,
            {"market_research": VALID_MARKET_QUERY_REWARD, "efficiency": STEP_PENALTY},
        )
        self._record_reward_components(
            market_research=VALID_MARKET_QUERY_REWARD,
            efficiency=-STEP_PENALTY,
        )
        self._mark_last_action_progress(True)
        return self._build_observation(
            step_reward=reward,
            message="Market conditions revealed.",
        )

    def _handle_flag_fraud(self, action: FinVerseAction) -> FinVerseObservation:
        reason = str(action.params.get("reason", "") or action.reasoning).strip()
        if not reason:
            return self._invalid("flag_fraud requires params.reason or reasoning.")
        if reason in self._fraud_flags:
            return self._invalid("This fraud flag has already been raised.", penalty=0.10)

        self._fraud_flags.append(reason)
        self._append_conversation("assistant", f"Fraud flag raised: {reason}")
        reward = VALID_FRAUD_FLAG_REWARD - STEP_PENALTY
        self._refresh_current_observation()
        self._cumulative_reward += reward
        self._set_last_info(
            "Fraud flag recorded for downstream review.",
            0.0,
            {"fraud_review": VALID_FRAUD_FLAG_REWARD, "efficiency": STEP_PENALTY},
        )
        self._record_reward_components(
            fraud_review=VALID_FRAUD_FLAG_REWARD,
            efficiency=-STEP_PENALTY,
        )
        self._mark_last_action_progress(True)
        return self._build_observation(
            step_reward=reward,
            message="Fraud flag recorded for downstream review.",
        )

    def _handle_terminal(self, action: FinVerseAction) -> FinVerseObservation:
        oracle_decision = str(self._ground_truth.get("decision", "deny")).lower()
        oracle_score = self._oracle_score_for_decision(action.action_type)
        explanation = self.oracle.explain_decision(
            self._applicant["features"],
            market_condition=self._market_state["name"],
        )
        confidence = float(self._ground_truth.get("confidence", 0.5))
        reward = -STEP_PENALTY
        penalties: Dict[str, float] = {"efficiency": STEP_PENALTY}

        if action.action_type == "escalate":
            reward += 0.0
            penalties["terminal_reward"] = 0.0
        else:
            correctness_reward = 1.0 if action.action_type == oracle_decision else -1.0
            confidence_bonus = 0.2 * confidence if action.action_type == oracle_decision else -0.2 * confidence
            fast_resolution_bonus = 0.2 if self._steps_taken <= 3 else 0.0
            reward += correctness_reward + confidence_bonus + fast_resolution_bonus
            penalties["terminal_reward"] = round(correctness_reward, 4)
            if confidence_bonus:
                penalties["confidence_bonus"] = round(confidence_bonus, 4)
            if fast_resolution_bonus:
                penalties["fast_resolution_bonus"] = round(fast_resolution_bonus, 4)

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
            penalties,
            oracle_decision=oracle_decision,
            oracle_confidence=confidence,
        )
        self._record_reward_components(
            correctness=penalties.get("terminal_reward", 0.0),
            confidence_bonus=penalties.get("confidence_bonus", 0.0),
            fast_resolution_bonus=penalties.get("fast_resolution_bonus", 0.0),
            efficiency=-STEP_PENALTY,
        )
        self._mark_last_action_progress(True)
        return self._build_observation(
            step_reward=reward,
            done=True,
            episode_score=reward,
            message=explanation["explanation"],
        )

    def step(self, action: str | FinVerseAction) -> Dict[str, Any]:
        active_error = self._ensure_active()
        if active_error is not None:
            return self._wrap_step_result(active_error)

        self._steps_taken += 1
        try:
            parsed_action = self._coerce_action(action)
        except ValueError as exc:
            invalid_observation = self._invalid(str(exc), penalty=0.25)
            if self._steps_taken >= MAX_STEPS:
                invalid_observation = self._finalize_timeout(invalid_observation)
            return self._wrap_step_result(invalid_observation)

        self._record_action(parsed_action)

        if parsed_action.action_type == "request_info":
            observation = self._handle_request_info(parsed_action)
        elif parsed_action.action_type == "query_market":
            observation = self._handle_query_market()
        elif parsed_action.action_type == "flag_fraud":
            observation = self._handle_flag_fraud(parsed_action)
        elif parsed_action.action_type in {"approve", "deny", "escalate"}:
            observation = self._handle_terminal(parsed_action)
        else:
            observation = self._invalid(f"Unsupported action type '{parsed_action.action_type}'.", penalty=0.25)

        repeat_penalty = self._apply_repeat_penalties(parsed_action)
        if repeat_penalty > 0.0:
            adjusted_reward = float(observation.step_reward) - repeat_penalty
            reward_components = dict(self._last_info.get("reward_components", {}))
            reward_components["anti_hacking"] = round(reward_components.get("anti_hacking", 0.0) - repeat_penalty, 4)
            self._last_info["reward_components"] = reward_components
            if observation.done:
                self._last_episode_score = adjusted_reward
                if self._auditor_compliance_log:
                    self._auditor_compliance_log[-1] = adjusted_reward
            observation = self._build_observation(
                step_reward=adjusted_reward,
                done=observation.done,
                episode_score=self._last_episode_score if observation.done else 0.0,
                message=observation.message,
            )

        if not observation.done and self._steps_taken >= MAX_STEPS:
            observation = self._finalize_timeout(observation)

        return self._wrap_step_result(observation)

    def state(self) -> FinVerseState:
        return FinVerseState(
            session_id=self._session_id,
            episode_id=self._episode_id,
            task_difficulty=self._difficulty,
            applicant_ground_truth=dict(self._applicant.get("features", {})),
            revealed_fields=dict(self._revealed_fields),
            applicant_is_fraudulent=bool(self._applicant.get("is_adversarial", False)),
            market_state=dict(self._market_state) if self._market_visible else {},
            conversation=list(self._conversation),
            action_history=list(self.trajectory),
            fraud_flags=list(self._fraud_flags),
            requested_fields=list(self._requested_fields),
            market_queried=self._market_visible,
            fraud_checked=bool(self._fraud_flags),
            steps_taken=self._steps_taken,
            auditor_compliance_log=list(self._auditor_compliance_log),
            episode_count=self._episode_count,
        )
