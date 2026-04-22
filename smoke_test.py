from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def assert_ok(response, label: str) -> dict:
    if response.status_code != 200:
        raise AssertionError(f"{label} failed: {response.status_code} {response.text[:300]}")
    return response.json()


def run_multistep_episode(task_name: str) -> None:
    reset_payload = assert_ok(client.post("/reset", json={"task_name": task_name}), "reset")
    assert reset_payload["done"] is False
    assert reset_payload["step"] == 0
    assert reset_payload["action_history"] == []

    request_field = reset_payload["applicant"]["missing_fields"][0]
    first_step = assert_ok(
        client.post("/step", json={"action_type": "request_info", "params": {"field": request_field}}),
        "request_info",
    )
    assert first_step["done"] is False
    assert first_step["reward"] > 0.0
    assert first_step["observation"]["step"] == 1
    assert len(first_step["observation"]["action_history"]) == 1

    second_step = assert_ok(client.post("/step", json={"action_type": "query_market"}), "query_market")
    assert second_step["done"] is False
    assert second_step["observation"]["market_visible"] is True
    assert len(second_step["observation"]["action_history"]) == 2

    final_step = assert_ok(
        client.post(
            "/step",
            json={"action_type": "approve", "reasoning": "Approving after reviewing applicant and market data."},
        ),
        "approve",
    )
    assert final_step["done"] is True
    assert final_step["observation"]["done"] is True
    assert len(final_step["observation"]["action_history"]) == 3


def run_timeout_episode(task_name: str) -> None:
    reset_payload = assert_ok(client.post("/reset", json={"task_name": task_name}), "reset-timeout")
    first_missing = reset_payload["applicant"]["missing_fields"][0]

    for step_index in range(1, 8):
        payload = {"action_type": "request_info", "params": {"field": first_missing}}
        result = assert_ok(client.post("/step", json=payload), f"timeout-step-{step_index}")
        if step_index < 8:
            assert result["observation"]["step"] == step_index
        if step_index < 8 and step_index < result["observation"]["max_steps"]:
            assert result["done"] is False

    timeout_result = assert_ok(client.post("/step", json={"action_type": "query_market"}), "timeout-final")
    assert timeout_result["done"] is True
    assert "timeout" in timeout_result["info"].get("penalties_applied", {})


for task in ("binary_decision", "risk_tiering", "adaptive_inquiry"):
    run_multistep_episode(task)

run_timeout_episode("binary_decision")

print("Smoke tests passed")
