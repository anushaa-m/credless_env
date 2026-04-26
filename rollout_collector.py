"""
rollout_collector.py

Runs episodes, calls the policy, triggers learning updates.
This is the only place where Agent2 decisions and updates happen.
No monkey-patching. No recursion risk.
"""

import json
import statistics
from pathlib import Path
from typing import Any

from agent2_policy import Agent2Policy


def _risk_score_from_agent(agent1: Any, obs: dict, env: Any | None = None) -> float:
    if hasattr(agent1, "score"):
        return float(agent1.score(obs))
    if "risk_score" in obs:
        return float(obs["risk_score"])
    if hasattr(agent1, "predict"):
        features = obs.get("features")
        if features is None and env is not None and hasattr(env, "current_feature_snapshot"):
            features = env.current_feature_snapshot()
        if features is None:
            raise ValueError("Agent1.predict needs observation features.")
        return float(agent1.predict(features))
    raise TypeError("agent1 must expose score(obs) or predict(features).")


def _normalize_oracle(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "reject":
        return "deny"
    return normalized


def run_episode(env, agent1, policy: Agent2Policy, task: str = "binary_decision") -> dict:
    """
    Single episode:
      1. reset env
      2. agent1 produces risk_score
      3. policy samples action
      4. env.step() returns reward and oracle decision
      5. policy.update() performs gradient step
    """
    obs = env.reset(task_name=task)

    risk_score = _risk_score_from_agent(agent1, obs, env)

    action_str, log_prob = policy.sample_action(risk_score)
    env_action = {
        "action_type": action_str,
        "params": {},
        "reasoning": f"p_approve={policy.p_approve(risk_score):.3f}, risk={risk_score:.3f}",
    }

    result = env.step(env_action)
    reward = float(result["reward"])
    info = result.get("info", {})
    oracle = info.get("oracle_decision", "UNKNOWN")

    update_info = policy.update(risk_score, action_str, reward)

    return {
        "risk_score": round(risk_score, 4),
        "action": action_str,
        "oracle_decision": oracle,
        "reward": round(reward, 4),
        "correct": _normalize_oracle(action_str) == _normalize_oracle(oracle),
        "p_approve": update_info["p_approve"],
        "weight": update_info["weight"],
        "bias": update_info["bias"],
        "advantage": update_info["advantage"],
        "log_prob": round(float(log_prob), 6),
    }


def run_training_loop(
    env,
    agent1,
    policy: Agent2Policy,
    n_episodes: int = 200,
    log_path: str = "rl/reward_log.jsonl",
    checkpoint_every: int = 25,
    print_every: int = 10,
):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    reward_window: list[float] = []
    approve_window: list[float] = []

    for ep in range(1, n_episodes + 1):
        ep_data = run_episode(env, agent1, policy)
        ep_data["episode"] = ep

        log_file.write(json.dumps(ep_data) + "\n")
        log_file.flush()

        reward_window.append(ep_data["reward"])
        approve_window.append(1.0 if ep_data["action"] == "approve" else 0.0)
        if len(reward_window) > 20:
            reward_window.pop(0)
            approve_window.pop(0)

        if ep % print_every == 0:
            avg_r = sum(reward_window) / len(reward_window)
            std_r = statistics.stdev(reward_window) if len(reward_window) > 1 else 0.0
            app_rate = sum(approve_window) / len(approve_window)
            print(
                f"Ep {ep:4d} | "
                f"reward={ep_data['reward']:+.3f} | "
                f"avg20={avg_r:+.3f} | "
                f"std20={std_r:.3f} | "
                f"approve_rate={app_rate:.2f} | "
                f"w={ep_data['weight']:+.3f} b={ep_data['bias']:+.3f}"
            )

        if ep % checkpoint_every == 0:
            policy.save()
            print(f"  [checkpoint saved at episode {ep}]")

    log_file.close()
    policy.save()
    print("\nTraining complete.")
