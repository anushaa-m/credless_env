import unittest

from inference import _aggregate_metrics


class InferenceMetricsTests(unittest.TestCase):
    def test_aggregate_metrics_report_reward_and_oracle_agreement(self):
        summary = _aggregate_metrics(
            [
                {"reward": 0.8, "decision": "APPROVE", "oracle_decision": "APPROVE"},
                {"reward": 0.2, "decision": "DENY", "oracle_decision": "REJECT"},
                {"reward": 0.4, "decision": "APPROVE", "oracle_decision": "DENY"},
            ]
        )

        self.assertEqual(summary["mean_reward"], 0.466667)
        self.assertEqual(summary["oracle_agreement"], 0.666667)
        self.assertEqual(summary["oracle_agreement_count"], 2)
        self.assertEqual(summary["episodes"], 3)


if __name__ == "__main__":
    unittest.main()
