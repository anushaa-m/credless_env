from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import FinVerseAction, FinVerseObservation, FinVerseState
from .data_generator import FIELD_NAMES, generate_applicant
from .graders import MIN_SCORE, audit_terminal_action, evaluate_terminal_action
from .oracle import MARKET_SCENARIOS, CredLessOracle
from .tasks import TASK_DIFFICULTY, TASK_NAMES

MAX_STEPS = 8
POLICY_REQUIRED_FIELDS = {
    "easy": ["payment_reliability", "debt_burden_score"],
    "medium": ["payment_reliability", "debt_burden_score", "overdraft_risk"],
    "hard": ["payment_reliability", "debt_burden_score", "overdraft_risk", "total_delinquency_score"],
}


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
        self._conversation: List[Dict[str, Any]] = []
        self._fraud_flags: List[str] = []
        self._requested_fields: List[str] = []
        self._revealed_fields: Dict[str, Dict[str, Any]] = {}
        self._done = False
        self._last_episode_score = 0.0

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

        return self._build_observation(
            message=(
                "Partial applicant profile received. Use request_info to ask for specific fields, "
                "query_market to reveal market conditions, flag_fraud when needed, and finish with "
                "approve, deny, or escalate."
            )
        )

    def _invalid(self, message: str, reward: float = -0.08) -> FinVerseObservation:
        self._cumulative_reward += reward
        return self._build_observation(step_reward=reward, message=message)

    def _ensure_active(self) -> Optional[FinVerseObservation]:
        if not self._episode_id:
            return FinVerseObservation(
                step_reward=-1.0,
                done=True,
                message="ERROR: call /reset before /step.",
                episode_score=MIN_SCORE,
                task_name=self._task,
            )
        if self._done:
            return self._build_observation(
                step_reward=0.0,
                done=True,
                episode_score=self._last_episode_score,
                message="Episode already completed. Call /reset to start a new episode.",
            )
        return None

    def _handle_request_info(self, action: FinVerseAction) -> FinVerseObservation:
        field = str(action.params.get("field", "")).strip()
        if not field:
            return self._invalid("request_info requires params.field.")
        if field not in FIELD_NAMES:
            return self._invalid(f"Unknown applicant field '{field}'.")
        if field in self._revealed_fields:
            return self._invalid(f"Field '{field}' is already visible.", reward=-0.05)

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

        reward = 0.02 if field in self._current_policy.get("required_fields", []) else 0.01
        self._cumulative_reward += reward
        return self._build_observation(
            step_reward=reward,
            message=f"Revealed '{field}' from applicant response.",
        )

    def _handle_query_market(self) -> FinVerseObservation:
        if self._market_visible:
            return self._invalid("Market state is already visible.", reward=-0.04)

        self._market_visible = True
        self._append_conversation("assistant", "Requesting current lending market conditions.")
        self._append_conversation(
            "system",
            (
                f"Market revealed: {self._market_state['name']} with base rate "
                f"{self._market_state['base_rate']} and outlook '{self._market_state['sector_outlook']}'."
            ),
        )
        reward = -0.03
        self._cumulative_reward += reward
        return self._build_observation(
            step_reward=reward,
            message="Market conditions revealed at the cost of one investigation step.",
        )

    def _handle_flag_fraud(self, action: FinVerseAction) -> FinVerseObservation:
        reason = str(action.params.get("reason", "") or action.reasoning).strip()
        if not reason:
            return self._invalid("flag_fraud requires params.reason or reasoning.")
        if reason in self._fraud_flags:
            return self._invalid("This fraud flag has already been raised.", reward=-0.04)

        self._fraud_flags.append(reason)
        self._append_conversation("assistant", f"Fraud flag raised: {reason}")

        suspicious = self._applicant.get("is_adversarial", False)
        reward = 0.06 if suspicious else -0.04
        self._cumulative_reward += reward
        return self._build_observation(
            step_reward=reward,
            message="Fraud flag recorded for downstream review.",
        )

    def _terminal_payload(self, action: FinVerseAction) -> Dict[str, Any]:
        params = dict(action.params)
        market_adjustment = float(self._market_state.get("default_risk_index", 1.0))
        expected_rate = 10.5
        if self._ground_truth.get("tier") == "medium_risk":
            expected_rate = 14.0
        elif self._ground_truth.get("tier") == "high_risk":
            expected_rate = 18.5
        expected_rate += (market_adjustment - 1.0) * 5.0

        return {
            "action_type": action.action_type,
            "decision": action.action_type if action.action_type in {"approve", "deny"} else params.get("decision", action.action_type),
            "tier": params.get("tier"),
            "rate": params.get("rate"),
            "reasoning": action.reasoning.strip(),
            "expected_rate": round(expected_rate, 2),
        }

    def _handle_terminal(self, action: FinVerseAction) -> FinVerseObservation:
        if action.action_type in {"approve", "deny"} and not action.reasoning.strip():
            return self._invalid(f"{action.action_type} requires reasoning.", reward=-0.12)
        if action.action_type == "escalate" and not action.reasoning.strip():
            return self._invalid("escalate requires reasoning.", reward=-0.12)

        final_action = self._terminal_payload(action)
        auditor_result = audit_terminal_action(
            final_action=final_action,
            oracle_truth=self._ground_truth,
            revealed_fields=self._revealed_fields,
            market_visible=self._market_visible,
            fraud_flags=self._fraud_flags,
        )
        result = evaluate_terminal_action(
            final_action=final_action,
            oracle_truth=self._ground_truth,
            auditor_result=auditor_result,
            requests_made=len(self._requested_fields),
            queried_market=self._market_visible,
            fraud_flags=self._fraud_flags,
            applicant_is_fraudulent=bool(self._applicant.get("is_adversarial", False)),
        )

        reward = float(result["reward"])
        self._cumulative_reward += reward
        self._done = True
        self._last_episode_score = float(result["episode_score"])
        self._auditor_compliance_log.append(float(auditor_result["score"]))
        self._append_conversation(
            "assistant",
            f"Final action: {action.action_type}. Reasoning: {action.reasoning.strip()}",
        )
        self._append_conversation(
            "system",
            (
                f"Episode resolved with oracle_decision={self._ground_truth['decision']}, "
                f"auditor_score={auditor_result['score']:.2f}, episode_score={result['episode_score']:.2f}."
            ),
        )

        return self._build_observation(
            step_reward=reward,
            done=True,
            episode_score=float(result["episode_score"]),
            message=(
                f"Decision={final_action['decision']}, oracle={self._ground_truth['decision']}, "
                f"task_score={result['task_score']:.2f}, auditor={result['auditor_score']:.2f}, "
                f"penalty={result['efficiency_penalty']:.2f}, fraud_bonus={result['fraud_bonus']:.2f}."
            ),
        )

    def step(self, action: FinVerseAction) -> FinVerseObservation:
        guard = self._ensure_active()
        if guard is not None:
            return guard

        self._steps_taken += 1
        if self._steps_taken > MAX_STEPS:
            self._done = True
            self._cumulative_reward -= 0.5
            self._last_episode_score = MIN_SCORE
            self._append_conversation("system", "Episode timed out before a terminal action was submitted.")
            return self._build_observation(
                step_reward=-0.5,
                done=True,
                episode_score=MIN_SCORE,
                message="Episode timed out before the investigation concluded.",
            )

        if action.action_type == "request_info":
            return self._handle_request_info(action)
        if action.action_type == "query_market":
            return self._handle_query_market()
        if action.action_type == "flag_fraud":
            return self._handle_flag_fraud(action)
        if action.action_type in {"approve", "deny", "escalate"}:
            return self._handle_terminal(action)
        return self._invalid(f"Unsupported action_type '{action.action_type}'.")

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
