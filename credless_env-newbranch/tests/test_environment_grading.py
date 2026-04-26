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

    def test_terminal_reward_records_simple_components(self):
        env = build_env()

        observation = env.step(
            FinVerseAction(
                action_type="approve",
                reasoning="Approve because payment reliability and debt burden score look strong in current market.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )

        self.assertTrue(observation["done"])
        self.assertIn("efficiency", env._last_info["penalties_applied"])
        self.assertIn("oracle_match", env._last_info["reward_components"])
        self.assertIn("efficiency", env._last_info["reward_components"])

    def test_terminal_reward_prefers_oracle_alignment(self):
        aligned = build_env()
        misaligned = build_env()

        aligned_obs = aligned.step(
            FinVerseAction(
                action_type="approve",
                reasoning="Approve.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )
        misaligned._ground_truth["decision"] = "deny"
        misaligned_obs = misaligned.step(
            FinVerseAction(
                action_type="approve",
                reasoning="Approve.",
                params={"tier": "low_risk", "rate": 10.0},
            )
        )

        self.assertGreater(float(aligned_obs["reward"]), float(misaligned_obs["reward"]))


if __name__ == "__main__":
    unittest.main()
