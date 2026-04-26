from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_SKIP = (
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("numpy") is None
)

if not _SKIP:
    import numpy as np

    from env.env_runner import (
        DISCRETE_ACTIONS,
        FEATURE_DIM,
        PPORolloutBuffer,
        PPOPolicy,
        RolloutStep,
        action_mask,
        discrete_to_env_action,
        observation_to_vector,
    )


def sample_observation():
    return {
        "applicant": {
            "profile": {
                "payment_reliability": {"value": 0.8, "confidence": 0.9},
                "debt_burden_score": {"value": 0.3, "confidence": 0.7},
            },
            "missing_fields": ["total_delinquency_score", "overdraft_risk", "medical_stress_score"],
        },
        "current_policy": {"required_fields": ["payment_reliability", "total_delinquency_score"]},
        "market_visible": False,
        "fraud_flags_raised": [],
        "fraud_checked": False,
        "market_state": {"default_risk_index": 1.05, "base_rate": 11.0},
        "oracle_risk": 0.4,
        "oracle_confidence": 0.6,
        "step": 1,
        "max_steps": 8,
    }


@unittest.skipIf(_SKIP, "torch and numpy required for PPO env runner tests")
class EnvRunnerPpoTests(unittest.TestCase):
    def test_observation_vector_shape(self):
        vector = observation_to_vector(sample_observation())
        self.assertEqual(vector.shape, (FEATURE_DIM,))

    def test_action_mask_limits_unavailable_actions(self):
        mask = action_mask(sample_observation())
        request_idx = DISCRETE_ACTIONS.index("request_info:total_delinquency_score")
        market_idx = DISCRETE_ACTIONS.index("query_market")
        approve_idx = DISCRETE_ACTIONS.index("approve")
        self.assertTrue(mask[request_idx])
        self.assertTrue(mask[market_idx])
        self.assertTrue(mask[approve_idx])

    def test_discrete_to_env_action_maps_request(self):
        idx = DISCRETE_ACTIONS.index("request_info:overdraft_risk")
        action = discrete_to_env_action(idx, sample_observation())
        self.assertEqual(action["action_type"], "request_info")
        self.assertEqual(action["params"]["field"], "overdraft_risk")

    def test_rollout_buffer_advantages_shape(self):
        buffer = PPORolloutBuffer()
        state = np.zeros(FEATURE_DIM, dtype=np.float64)
        mask = np.ones(len(DISCRETE_ACTIONS), dtype=np.float64)
        buffer.add(RolloutStep(state, 0, -0.1, 0.2, False, 0.05, mask))
        buffer.add(RolloutStep(state, 1, -0.2, 0.4, True, 0.10, mask))
        returns, advantages = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
        self.assertEqual(returns.shape, (2,))
        self.assertEqual(advantages.shape, (2,))

    def test_policy_update_and_save_load(self):
        policy = PPOPolicy(seed=7)
        states = np.stack([observation_to_vector(sample_observation()) for _ in range(4)])
        actions = np.asarray([0, 1, 2, 3], dtype=np.int64)
        old_log_probs = np.asarray([-0.5, -0.6, -0.4, -0.7], dtype=np.float64)
        returns = np.asarray([0.2, 0.4, 0.1, 0.3], dtype=np.float64)
        advantages = np.asarray([0.1, -0.2, 0.3, 0.05], dtype=np.float64)
        masks = np.ones((4, len(DISCRETE_ACTIONS)), dtype=np.float64)

        metrics = policy.update(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            returns=returns,
            advantages=advantages,
            masks=masks,
            lr=0.01,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            epochs=2,
        )

        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)

        with TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "ppo.npz"
            policy.save(weights_path)
            restored = PPOPolicy.load(weights_path)
            self.assertEqual(restored.policy_w.shape, policy.policy_w.shape)
            self.assertEqual(restored.value_w.shape, policy.value_w.shape)


if __name__ == "__main__":
    unittest.main()
