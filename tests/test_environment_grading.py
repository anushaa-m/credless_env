import unittest
from unittest.mock import Mock

from models import FinVerseAction
from server.data_generator import generate_applicant
from server.environment import CreditAnalystEnvironment


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


if __name__ == "__main__":
    unittest.main()
