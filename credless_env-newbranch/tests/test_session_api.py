import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import server.app as app_module


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = app_module.SessionStore(Path(self.temp_dir.name) / "sessions")
        self.store_patch = patch.object(app_module, "store", self.store)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        self.client = TestClient(app_module.app)

    def _reset(self, **payload):
        response = self.client.post("/reset", json=payload)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _step(self, session_id: str, episode_id: str, action_type: str = "query_market"):
        response = self.client.post(
            "/step",
            json={
                "session_id": session_id,
                "episode_id": episode_id,
                "action": {"action_type": action_type, "params": {}, "reasoning": "test"},
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_reset_returns_session_and_episode_ids(self):
        payload = self._reset()

        self.assertTrue(payload["session_id"])
        self.assertTrue(payload["episode_id"])
        self.assertEqual(payload["info"]["session_id"], payload["session_id"])
        self.assertEqual(payload["info"]["episode_id"], payload["episode_id"])

    def test_reset_returns_storytelling_factors(self):
        payload = self._reset(seed=1)

        self.assertIn("top_factors", payload)
        self.assertIn("top_factors", payload["observation"])
        self.assertIn("top_factors", payload["info"])
        self.assertIsInstance(payload["top_factors"], list)
        self.assertGreater(len(payload["top_factors"]), 0)
        self.assertIsInstance(payload["oracle_confidence"], float)

    def test_step_persists_state_for_session(self):
        payload = self._reset()

        self._step(payload["session_id"], payload["episode_id"])
        state = self.client.get("/state", params={"session_id": payload["session_id"]})

        self.assertEqual(state.status_code, 200)
        state_payload = state.json()
        self.assertEqual(state_payload["session_id"], payload["session_id"])
        self.assertEqual(state_payload["episode_id"], payload["episode_id"])
        self.assertEqual(state_payload["steps_taken"], 1)

    def test_sessions_are_isolated(self):
        first = self._reset()
        second = self._reset()

        self._step(first["session_id"], first["episode_id"])
        first_state = self.client.get("/state", params={"session_id": first["session_id"]}).json()
        second_state = self.client.get("/state", params={"session_id": second["session_id"]}).json()

        self.assertEqual(first_state["steps_taken"], 1)
        self.assertEqual(second_state["steps_taken"], 0)
        self.assertNotEqual(first_state["session_id"], second_state["session_id"])

    def test_step_rejects_stale_episode_id(self):
        first = self._reset()
        second = self._reset(session_id=first["session_id"])

        response = self.client.post(
            "/step",
            json={
                "session_id": first["session_id"],
                "episode_id": first["episode_id"],
                "action": {"action_type": "query_market", "params": {}, "reasoning": "stale"},
            },
        )

        self.assertEqual(response.status_code, 409)
        payload = response.json()["detail"]
        self.assertEqual(payload["session_id"], first["session_id"])
        self.assertEqual(payload["expected_episode_id"], second["episode_id"])
        self.assertEqual(payload["received_episode_id"], first["episode_id"])


if __name__ == "__main__":
    unittest.main()
