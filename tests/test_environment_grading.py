import importlib.util
import unittest
from unittest.mock import Mock
import sys
import types

try:
    from models import FinVerseAction
except ModuleNotFoundError as exc:
    # Test-only compatibility shim: allow unit tests without openenv installed.
    if exc.name not in {"openenv", "pydantic"}:
        raise

    pydantic_mod = types.ModuleType("pydantic")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

    def _Field(*, default=None, default_factory=None, **_kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = _Field
    sys.modules["pydantic"] = pydantic_mod

    openenv_mod = types.ModuleType("openenv")
    core_mod = types.ModuleType("openenv.core")
    env_server_mod = types.ModuleType("openenv.core.env_server")
    types_mod = types.ModuleType("openenv.core.env_server.types")
    interfaces_mod = types.ModuleType("openenv.core.env_server.interfaces")
    client_types_mod = types.ModuleType("openenv.core.client_types")
    env_client_mod = types.ModuleType("openenv.core.env_client")

    class _Action:
        pass

    class _Observation:
        pass

    class _State:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _Environment:
        pass

    class _StepResult:
        pass

    class _EnvClient:
        def __class_getitem__(cls, _item):
            return cls

    types_mod.Action = _Action
    types_mod.Observation = _Observation
    types_mod.State = _State
    interfaces_mod.Environment = _Environment
    client_types_mod.StepResult = _StepResult
    env_client_mod.EnvClient = _EnvClient

    sys.modules["openenv"] = openenv_mod
    sys.modules["openenv.core"] = core_mod
    sys.modules["openenv.core.env_server"] = env_server_mod
    sys.modules["openenv.core.env_server.types"] = types_mod
    sys.modules["openenv.core.env_server.interfaces"] = interfaces_mod
    sys.modules["openenv.core.client_types"] = client_types_mod
    sys.modules["openenv.core.env_client"] = env_client_mod

    from models import FinVerseAction
_missing_runtime_deps = [
    dep
    for dep in ("numpy", "pandas", "sklearn")
    if importlib.util.find_spec(dep) is None
]
_dependency_error = None

try:
    if _missing_runtime_deps:
        raise ModuleNotFoundError(f"Missing runtime deps: {', '.join(_missing_runtime_deps)}")
    from server.data_generator import generate_applicant
    from server.environment import CreditAnalystEnvironment
except ModuleNotFoundError as dep_exc:
    _dependency_error = dep_exc

    def generate_applicant(*_args, **_kwargs):
        raise RuntimeError(str(_dependency_error))

    class CreditAnalystEnvironment:  # type: ignore[override]
        pass


def build_env() -> CreditAnalystEnvironment:
    env = CreditAnalystEnvironment()
    env.oracle.explain_decision = Mock(return_value={"explanation": "oracle explanation"})
    env._applicant = {
        "features": {"payment_reliability": 0.8},
        "is_adversarial": False,
    }
    env._market_state = {
        "name": "Stable Credit",
        "base_rate": 10.0,
        "default_risk_index": 1.0,
        "sector_outlook": "stable",
    }
    env._ground_truth = {
        "decision": "approve",
        "tier": "low_risk",
        "confidence": 0.82,
        "default_prob": 0.2,
    }
    env._revealed_fields = {
        "payment_reliability": {"value": 0.8, "confidence": 0.9},
        "debt_burden_score": {"value": 0.2, "confidence": 0.8},
    }
    env._market_visible = True
    env._requested_fields = ["payment_reliability", "debt_burden_score"]
    env._fraud_flags = []
    env._steps_taken = 2
    env._episode_id = "episode-1"
    return env


@unittest.skipIf(_dependency_error is not None, f"Environment tests skipped: {_dependency_error}")
class EnvironmentGradingTests(unittest.TestCase):
    def test_deception_level_marks_applicant_metadata(self):
        applicant = generate_applicant(seed=7, difficulty="hard", deception_level=1.0)

        self.assertTrue(applicant["is_adversarial"])
        self.assertEqual(applicant["deception_level"], 1.0)
        self.assertEqual(applicant["data_quality"], "self_reported")
        self.assertIn("transaction_health", applicant["fabricated_fields"])
        self.assertTrue(applicant["withheld_fields"])

    def test_fraud_flag_rewards_matched_deception(self):
        env = CreditAnalystEnvironment()
        env.reset(task_name="adaptive_inquiry", seed=7, deception_level=1.0)

        observation = env.step(
            {
                "action_type": "flag_fraud",
                "params": {"reason": "income and transaction confidence look inconsistent"},
                "reasoning": "income and transaction confidence look inconsistent",
            }
        )

        self.assertGreater(observation["reward"], 0.05)
        self.assertTrue(observation["info"]["fraud_signal_match"])

    def test_terminal_reward_records_auditor_components(self):
        env = build_env()

        observation = env._handle_terminal(
            FinVerseAction(
                action_type="approve",
                reasoning="Approve because payment reliability and debt burden score look strong in current market.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )

        self.assertTrue(observation.done)
        self.assertIn("auditor_score", env._last_info["penalties_applied"])
        self.assertIn("task_score", env._last_info["penalties_applied"])
        self.assertIn("reasoning_score", env._last_info["penalties_applied"])
        self.assertIn("oracle_alignment", env._last_info["reward_components"])

    def test_reasoning_quality_changes_terminal_reward(self):
        weak_env = build_env()
        strong_env = build_env()

        weak = weak_env._handle_terminal(
            FinVerseAction(
                action_type="approve",
                reasoning="approve",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )
        strong = strong_env._handle_terminal(
            FinVerseAction(
                action_type="approve",
                reasoning="Approve because payment reliability and debt burden score support low risk in current market conditions.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )

        self.assertGreater(strong.step_reward, weak.step_reward)

    def test_conditional_approve_rewarded_for_medium_risk(self):
        env = build_env()
        env._ground_truth.update(
            {
                "decision": "approve",
                "tier": "medium_risk",
                "default_prob": 0.34,
                "recommended_action": "conditional_approve",
                "conditional_candidate": True,
            }
        )
        env.oracle.explain_decision = Mock(
            return_value={
                "explanation": "medium risk with overdraft pressure",
                "feature_contributions": [("overdraft_risk", 0.21), ("debt_burden_score", 0.18)],
            }
        )

        observation = env._handle_terminal(
            FinVerseAction(
                action_type="conditional_approve",
                reasoning="Conditional approval with elevated rate due to overdraft risk.",
                params={"rate": 14.5, "max_amount": 50000},
            )
        )
        self.assertTrue(observation.done)
        self.assertGreater(observation.step_reward, 0.30)

    def test_conditional_approve_penalized_for_high_risk(self):
        env = build_env()
        env._ground_truth.update(
            {
                "decision": "deny",
                "tier": "high_risk",
                "default_prob": 0.81,
                "recommended_action": "deny",
                "conditional_candidate": False,
            }
        )
        env.oracle.explain_decision = Mock(
            return_value={
                "explanation": "high risk profile",
                "feature_contributions": [("total_delinquency_score", 0.35)],
            }
        )

        observation = env._handle_terminal(
            FinVerseAction(
                action_type="conditional_approve",
                reasoning="Conditional terms despite high delinquency.",
                params={"rate": 14.5, "max_amount": 25000},
            )
        )
        self.assertTrue(observation.done)
        self.assertLess(observation.step_reward, 0.0)

    def test_income_lie_detection_reward_for_gt_2sigma(self):
        env = build_env()
        env._applicant.update(
            {
                "is_adversarial": True,
                "fabricated_fields": ["transaction_health"],
                "withheld_fields": [],
                "income_verification": {
                    "stated_income": 120000.0,
                    "inferred_income": 42000.0,
                    "sigma_scale": 15000.0,
                    "discrepancy_sigma": 2.6,
                    "is_income_lie": True,
                },
            }
        )
        env._revealed_fields.update(
            {
                "stated_income": {"value": 120000.0, "confidence": 0.86},
                "transaction_health": {"value": 0.25, "confidence": 0.8},
            }
        )

        observation = env._handle_verify_income(
            FinVerseAction(
                action_type="verify_income",
                reasoning="Income verification found >2 sigma discrepancy vs transaction behavior.",
                params={"note": "income mismatch over threshold"},
            )
        )

        self.assertGreater(observation.step_reward, 0.45)
        self.assertTrue(env._income_lie_detected)
        self.assertTrue(env._last_info["income_lie_detected"])

    def test_income_lie_false_positive_penalty(self):
        env = build_env()
        env._applicant.update(
            {
                "is_adversarial": False,
                "fabricated_fields": [],
                "withheld_fields": [],
                "income_verification": {
                    "stated_income": 52000.0,
                    "inferred_income": 50000.0,
                    "sigma_scale": 12000.0,
                    "discrepancy_sigma": 0.17,
                    "is_income_lie": False,
                },
            }
        )
        env._revealed_fields.update(
            {
                "stated_income": {"value": 52000.0, "confidence": 0.86},
                "transaction_health": {"value": 0.61, "confidence": 0.85},
            }
        )

        observation = env._handle_verify_income(
            FinVerseAction(
                action_type="verify_income",
                reasoning="Suspicious income profile",
                params={"note": "check income"},
            )
        )

        self.assertLess(observation.step_reward, -0.25)
        self.assertFalse(env._income_lie_detected)

    def test_missed_income_lie_penalty_on_approve(self):
        env = build_env()
        env._applicant.update(
            {
                "is_adversarial": True,
                "fabricated_fields": ["transaction_health"],
                "withheld_fields": [],
                "income_verification": {
                    "stated_income": 110000.0,
                    "inferred_income": 38000.0,
                    "sigma_scale": 14000.0,
                    "discrepancy_sigma": 2.4,
                    "is_income_lie": True,
                },
            }
        )
        env._revealed_fields.update(
            {
                "stated_income": {"value": 110000.0, "confidence": 0.86},
                "transaction_health": {"value": 0.22, "confidence": 0.81},
            }
        )
        env.oracle.explain_decision = Mock(
            return_value={
                "explanation": "approve path",
                "feature_contributions": [("payment_reliability", -0.11)],
            }
        )

        observation = env._handle_terminal(
            FinVerseAction(
                action_type="approve",
                reasoning="Approving based on observed profile.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )

        self.assertTrue(observation.done)
        self.assertLess(observation.step_reward, 0.0)
        self.assertEqual(env._last_info["penalties_applied"]["missed_income_lie"], 1.0)


if __name__ == "__main__":
    unittest.main()
