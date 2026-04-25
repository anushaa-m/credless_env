"""
Environment interaction runner plus lightweight PPO trainer.

Supports:
  - heuristic or model-driven rollouts
  - session-safe environment interaction
  - numpy PPO training loop over discrete environment actions
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
import re
import torch

load_dotenv()

# ===== LLM POLICY SWITCH =====
USE_LLM_POLICY = True

# ===== LLM HELPER FUNCTIONS =====
def extract_json_safe(text):
    try:
        if "FINAL ANSWER:" in text:
            text = text.split("FINAL ANSWER:")[-1]

        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError

        return json.loads(match.group())
    except Exception:
        return {
            "action_type": "reject",
            "params": {},
            "reasoning": "fallback",
        }


def generate_action(model, tokenizer, observation):
    prompt = f"""
You are a credit decision agent.

Observation:
{observation}

Return ONLY JSON:

{{
  "action_type": "approve or reject",
  "params": {{}},
  "reasoning": "short"
}}

FINAL ANSWER:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    action = extract_json_safe(text)

    action.setdefault("params", {})
    action.setdefault("reasoning", "")
    action["action_type"] = action["action_type"].lower()

    return action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://anushaa-m-credless-env.hf.space")
BENCHMARK = os.getenv("CREDLESS_BENCHMARK", "credless-env")
TASKS = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
TASK_NAME = os.getenv("CREDLESS_TASK")
MAX_STEPS = 8
MIN_SCORE = 0.01
MAX_SCORE = 0.99
MODEL_DIR = Path(__file__).resolve().parent / "saved"
PPO_WEIGHTS_PATH = MODEL_DIR / "ppo_policy.npz"

REQUEST_PRIORITY = [
    "total_delinquency_score",
    "overdraft_risk",
    "medical_stress_score",
    "debt_burden_score",
    "payment_reliability",
    "income_capacity_score",
    "employment_stability",
    "account_maturity",
]

TERMINAL_ACTIONS = ["approve", "deny", "escalate"]
DISCRETE_ACTIONS = (
    [f"request_info:{field}" for field in REQUEST_PRIORITY]
    + ["query_market", "flag_fraud"]
    + TERMINAL_ACTIONS
)
FEATURE_DIM = len(REQUEST_PRIORITY) + 12

SYSTEM_PROMPT = (
    "You are a credit decision agent for a multi-step RL environment. "
    "Return a single JSON action matching the allowed schema."
)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None
model = None
tokenizer = None


def strict_score(value: float) -> float:
    return round(min(MAX_SCORE, max(MIN_SCORE, float(value))), 4)


def sanitize_log_value(value: str) -> str:
    return " ".join(str(value).split())


def format_action(action: dict) -> str:
    return sanitize_log_value(json.dumps(action))


def profile_values(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("value", 0.0)) for field, payload in profile.items()}


def profile_confidence(observation: Dict[str, object]) -> Dict[str, float]:
    profile = observation.get("applicant", {}).get("profile", {})
    return {field: float(payload.get("confidence", 0.0)) for field, payload in profile.items()}


def _estimate_default_risk(fields: Dict[str, float], market_index: float) -> float:
    risk = (
        0.16 * fields.get("revolving_utilization", 0.5)
        + 0.08 * fields.get("delinquency_30_59", 0.0)
        + 0.10 * fields.get("delinquency_60_89", 0.0)
        + 0.12 * fields.get("delinquency_90plus", 0.0)
        + 0.12 * fields.get("total_delinquency_score", 0.0)
        + 0.12 * fields.get("debt_burden_score", 0.0)
        + 0.08 * fields.get("medical_stress_score", 0.0)
        + 0.08 * fields.get("overdraft_risk", 0.0)
        + 0.06 * fields.get("location_risk_index", 0.0)
        + 0.08 * (1.0 - fields.get("payment_reliability", 0.5))
        + 0.07 * (1.0 - fields.get("income_capacity_score", 0.5))
        + 0.05 * (1.0 - fields.get("employment_stability", 0.5))
        + 0.04 * (1.0 - fields.get("account_maturity", 0.5))
    )
    risk -= 0.04 * fields.get("net_worth_score", 0.5)
    risk -= 0.02 * fields.get("asset_ownership_score", 0.5)
    risk *= market_index
    return max(0.0, min(1.0, risk))


def build_terminal_action(observation: Dict[str, object]) -> Dict[str, object]:
    fields = profile_values(observation)
    market_state = observation.get("market_state") or {}
    market_index = float(market_state.get("default_risk_index", 1.0))
    default_risk = _estimate_default_risk(fields, market_index)
    decision = "deny" if default_risk >= 0.50 else "approve"
    return {
        "action_type": decision,
        "reasoning": "Decision based on revealed field values and market context.",
    }


def policy_action(observation: Dict[str, object], task_name: str) -> Dict[str, object]:
    del task_name
    profile = observation.get("applicant", {}).get("profile", {})
    missing_fields = list(observation.get("applicant", {}).get("missing_fields", []))
    required_fields = list((observation.get("current_policy") or {}).get("required_fields", []))

    for field in required_fields:
        if field in missing_fields:
            return {"action_type": "request_info", "params": {"field": field}}

    for field in REQUEST_PRIORITY:
        if field in missing_fields and field not in profile:
            return {"action_type": "request_info", "params": {"field": field}}

    if not observation.get("market_visible", False):
        return {"action_type": "query_market"}

    confidence = profile_confidence(observation)
    suspicious = any((1.0 - value) > 0.25 for value in confidence.values())
    if suspicious and not observation.get("fraud_flags_raised"):
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "low-confidence profile values require manual verification"},
            "reasoning": "Low-confidence data suggests fraud review before decision.",
        }

    return build_terminal_action(observation)


def observation_to_vector(observation: Dict[str, object]) -> np.ndarray:
    profile = profile_values(observation)
    confidence = profile_confidence(observation)
    missing_fields = set(observation.get("applicant", {}).get("missing_fields", []))
    market_state = observation.get("market_state") or {}
    vector: List[float] = []

    for field in REQUEST_PRIORITY:
        vector.append(profile.get(field, 0.0))

    vector.extend(
        [
            float(len(profile)) / max(1.0, float(len(REQUEST_PRIORITY))),
            float(len(missing_fields)) / max(1.0, float(len(REQUEST_PRIORITY))),
            float(np.mean(list(confidence.values()))) if confidence else 0.0,
            float(np.min(list(confidence.values()))) if confidence else 0.0,
            float(observation.get("oracle_risk", 0.0)),
            float(observation.get("oracle_confidence", 0.0)),
            float(observation.get("market_visible", False)),
            float(observation.get("fraud_checked", False)),
            float(len(observation.get("fraud_flags_raised", [])) > 0),
            float(observation.get("step", 0)) / float(max(1, observation.get("max_steps", MAX_STEPS))),
            float(market_state.get("default_risk_index", 1.0)),
            float(market_state.get("base_rate", 0.0)) / 20.0,
        ]
    )
    return np.asarray(vector, dtype=np.float64)


def action_mask(observation: Dict[str, object]) -> np.ndarray:
    profile = observation.get("applicant", {}).get("profile", {})
    missing_fields = set(observation.get("applicant", {}).get("missing_fields", []))
    fraud_flags = list(observation.get("fraud_flags_raised", []))
    mask = np.zeros(len(DISCRETE_ACTIONS), dtype=bool)

    for idx, token in enumerate(DISCRETE_ACTIONS):
        if token.startswith("request_info:"):
            field = token.split(":", 1)[1]
            mask[idx] = field in missing_fields and field not in profile
        elif token == "query_market":
            mask[idx] = not bool(observation.get("market_visible", False))
        elif token == "flag_fraud":
            confidence = profile_confidence(observation)
            suspicious = any((1.0 - value) > 0.25 for value in confidence.values())
            mask[idx] = suspicious and not fraud_flags
        else:
            mask[idx] = True

    if not mask.any():
        mask[-3:] = True
    return mask


def discrete_to_env_action(action_idx: int, observation: Dict[str, object]) -> Dict[str, object]:
    token = DISCRETE_ACTIONS[action_idx]
    if token.startswith("request_info:"):
        return {"action_type": "request_info", "params": {"field": token.split(":", 1)[1]}}
    if token == "query_market":
        return {"action_type": "query_market"}
    if token == "flag_fraud":
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "ppo policy requested fraud review"},
            "reasoning": "Policy flagged suspicious confidence pattern.",
        }
    if token in TERMINAL_ACTIONS:
        return {"action_type": token, "reasoning": f"ppo policy selected {token}"}
    return build_terminal_action(observation)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = np.sum(exp)
    if total <= 0.0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / total


@dataclass
class StepLog:
    step: int
    action: str
    reward: float
    done: bool
    error: Optional[str]


@dataclass
class EpisodeResult:
    session_id: str
    episode_id: str
    total_reward: float
    final_score: float
    rewards: List[float]
    steps_taken: int


@dataclass
class RolloutStep:
    state: np.ndarray
    action_idx: int
    log_prob: float
    reward: float
    done: bool
    value: float
    mask: np.ndarray


class PPORolloutBuffer:
    def __init__(self):
        self.steps: List[RolloutStep] = []

    def add(self, step: RolloutStep) -> None:
        self.steps.append(step)

    def clear(self) -> None:
        self.steps.clear()

    def __len__(self) -> int:
        return len(self.steps)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray([step.reward for step in self.steps], dtype=np.float64)
        dones = np.asarray([step.done for step in self.steps], dtype=np.float64)
        values = np.asarray([step.value for step in self.steps], dtype=np.float64)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = 0.0
        for idx in range(len(self.steps) - 1, -1, -1):
            non_terminal = 1.0 - dones[idx]
            delta = rewards[idx] + gamma * next_value * non_terminal - values[idx]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[idx] = last_gae
            next_value = values[idx]
        returns = advantages + values
        return returns, advantages


class PPOPolicy:
    def __init__(self, feature_dim: int = FEATURE_DIM, action_dim: int = len(DISCRETE_ACTIONS), seed: int = 42):
        rng = np.random.default_rng(seed)
        self.policy_w = rng.normal(0.0, 0.02, size=(feature_dim, action_dim))
        self.policy_b = np.zeros(action_dim, dtype=np.float64)
        self.value_w = rng.normal(0.0, 0.02, size=(feature_dim,))
        self.value_b = 0.0

    def masked_probs(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        logits = state @ self.policy_w + self.policy_b
        masked_logits = np.where(mask, logits, -1e9)
        return softmax(masked_logits)

    def value(self, state: np.ndarray) -> float:
        return float(state @ self.value_w + self.value_b)

    def sample_action(self, state: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
        probs = self.masked_probs(state, mask)
        action_idx = int(rng.choice(len(probs), p=probs))
        log_prob = float(np.log(max(probs[action_idx], 1e-12)))
        return action_idx, log_prob, self.value(state)

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
        masks: np.ndarray,
        lr: float,
        clip_eps: float,
        entropy_coef: float,
        value_coef: float,
        epochs: int,
    ) -> dict[str, float]:
        norm_adv = advantages.copy()
        if len(norm_adv) > 1 and float(np.std(norm_adv)) > 1e-8:
            norm_adv = (norm_adv - np.mean(norm_adv)) / (np.std(norm_adv) + 1e-8)

        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0

        for _ in range(epochs):
            grad_policy_w = np.zeros_like(self.policy_w)
            grad_policy_b = np.zeros_like(self.policy_b)
            grad_value_w = np.zeros_like(self.value_w)
            grad_value_b = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0

            for idx, state in enumerate(states):
                mask = masks[idx].astype(bool)
                probs = self.masked_probs(state, mask)
                action = int(actions[idx])
                new_log_prob = float(np.log(max(probs[action], 1e-12)))
                ratio = float(np.exp(new_log_prob - old_log_probs[idx]))
                clipped_ratio = float(np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps))
                advantage = float(norm_adv[idx])
                use_clip = (advantage >= 0.0 and ratio > 1.0 + clip_eps) or (advantage < 0.0 and ratio < 1.0 - clip_eps)
                coeff = 0.0 if use_clip else ratio * advantage

                one_hot = np.zeros(len(DISCRETE_ACTIONS), dtype=np.float64)
                one_hot[action] = 1.0
                grad_logits = -(one_hot - probs) * coeff
                grad_policy_w += np.outer(state, grad_logits)
                grad_policy_b += grad_logits

                entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
                total_entropy += entropy
                entropy_grad = probs * (np.log(np.clip(probs, 1e-12, 1.0)) + 1.0)
                entropy_grad -= np.sum(entropy_grad) * probs
                grad_policy_w -= entropy_coef * np.outer(state, entropy_grad)
                grad_policy_b -= entropy_coef * entropy_grad

                value = self.value(state)
                value_error = value - float(returns[idx])
                grad_value_w += value_coef * 2.0 * value_error * state
                grad_value_b += value_coef * 2.0 * value_error

                unclipped = ratio * advantage
                clipped = clipped_ratio * advantage
                total_policy_loss += -min(unclipped, clipped)
                total_value_loss += value_error * value_error

            batch_size = max(1, len(states))
            self.policy_w -= lr * grad_policy_w / batch_size
            self.policy_b -= lr * grad_policy_b / batch_size
            self.value_w -= lr * grad_value_w / batch_size
            self.value_b -= lr * grad_value_b / batch_size

            mean_policy_loss = total_policy_loss / batch_size
            mean_value_loss = total_value_loss / batch_size
            mean_entropy = total_entropy / batch_size

        return {
            "policy_loss": float(mean_policy_loss),
            "value_loss": float(mean_value_loss),
            "entropy": float(mean_entropy),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            policy_w=self.policy_w,
            policy_b=self.policy_b,
            value_w=self.value_w,
            value_b=np.asarray([self.value_b], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "PPOPolicy":
        data = np.load(path)
        policy = cls(feature_dim=data["policy_w"].shape[0], action_dim=data["policy_w"].shape[1])
        policy.policy_w = data["policy_w"]
        policy.policy_b = data["policy_b"]
        policy.value_w = data["value_w"]
        policy.value_b = float(data["value_b"][0])
        return policy


class EnvironmentRunner:
    def __init__(self, env_base_url: str = ENV_BASE_URL, model_name: str = MODEL_NAME, model=None, tokenizer=None):
        self.env_base_url = env_base_url.rstrip("/")
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer

    def log_start(self, task: str) -> None:
        print(f"[START] task={task} env={BENCHMARK} model={self.model_name}", flush=True)

    def log_step(self, log: StepLog) -> None:
        error_val = log.error if log.error else "null"
        print(
            f"[STEP] step={log.step} action={log.action} reward={log.reward:.2f} "
            f"done={str(log.done).lower()} error={error_val}",
            flush=True,
        )

    def log_end(self, success: bool, steps: int, score: float, rewards: List[float]) -> None:
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )

    def call_model(self, observation: Dict[str, object], task_name: str) -> Dict[str, object]:
        if client is None:
            return policy_action(observation, task_name)

        try:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(observation)},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            content = (completion.choices[0].message.content or "").strip()
            action = json.loads(content)
            if not isinstance(action, dict) or "action_type" not in action:
                raise ValueError("Model response was not a valid action object")
            return action
        except Exception:
            return policy_action(observation, task_name)

    def reset_env(self, task_name: str, seed: Optional[int] = None, session_id: Optional[str] = None) -> Dict[str, object]:
        response = requests.post(
            f"{self.env_base_url}/reset",
            json={"task_name": task_name, "seed": seed, "session_id": session_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def step_env(self, session_id: str, episode_id: str, action: Dict[str, object]) -> Dict[str, object]:
        response = requests.post(
            f"{self.env_base_url}/step",
            json={"session_id": session_id, "episode_id": episode_id, "action": action},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def run_task(self, task_name: str) -> float:
        result = self.rollout(task_name=task_name, action_source="model")
        return result.final_score

    def rollout(
        self,
        task_name: str,
        *,
        action_source: str,
        ppo_policy: Optional[PPOPolicy] = None,
        seed: Optional[int] = None,
        log_steps: bool = True,
    ) -> EpisodeResult:
        rewards: List[float] = []
        steps_taken = 0
        final_reward = MIN_SCORE
        if log_steps:
            self.log_start(task_name)

        reset_payload = self.reset_env(task_name=task_name, seed=seed, session_id=str(uuid.uuid4()))
        observation = reset_payload.get("observation", reset_payload)
        session_id = str(reset_payload.get("session_id", ""))
        episode_id = str(reset_payload.get("episode_id", ""))
        done = bool(reset_payload.get("done", False))
        rng = np.random.default_rng(seed)

        try:
            while not done and steps_taken < MAX_STEPS:
                steps_taken += 1
                if action_source == "ppo":
                    if ppo_policy is None:
                        raise ValueError("ppo_policy required for PPO rollouts")
                    state = observation_to_vector(observation)
                    mask = action_mask(observation)
                    action_idx, _log_prob, _value = ppo_policy.sample_action(state, mask, rng)
                    action = discrete_to_env_action(action_idx, observation)
                else:
                    if USE_LLM_POLICY and self.model is not None:
                        action = generate_action(self.model, self.tokenizer, observation)
                    else:
                        action = self.call_model(observation, task_name)

                action_str = format_action(action)
                error = None
                try:
                    result = self.step_env(session_id, episode_id, action)
                    observation = result.get("observation", result)
                    reward = float(result.get("reward", 0.0))
                    final_reward = reward
                    done = bool(result.get("done", False))
                    episode_id = str(result.get("episode_id", episode_id))
                except Exception as exc:
                    reward = 0.0
                    final_reward = reward
                    done = True
                    error = sanitize_log_value(str(exc))

                rewards.append(reward)
                if log_steps:
                    self.log_step(StepLog(step=steps_taken, action=action_str, reward=reward, done=done, error=error))
        finally:
            final_score = strict_score(final_reward)
            if log_steps:
                self.log_end(final_score > 0.0, steps_taken, final_score, rewards)

        return EpisodeResult(
            session_id=session_id,
            episode_id=episode_id,
            total_reward=float(sum(rewards)),
            final_score=final_score,
            rewards=rewards,
            steps_taken=steps_taken,
        )


class PPOTrainer:
    def __init__(
        self,
        runner: EnvironmentRunner,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        learning_rate: float = 0.01,
        update_epochs: int = 5,
        seed: int = 42,
    ):
        self.runner = runner
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        self.rng = np.random.default_rng(seed)
        self.policy = PPOPolicy(seed=seed)

    def collect_episode(self, task_name: str, seed: Optional[int] = None) -> tuple[PPORolloutBuffer, EpisodeResult]:
        buffer = PPORolloutBuffer()
        reset_payload = self.runner.reset_env(task_name=task_name, seed=seed, session_id=str(uuid.uuid4()))
        observation = reset_payload.get("observation", reset_payload)
        session_id = str(reset_payload.get("session_id", ""))
        episode_id = str(reset_payload.get("episode_id", ""))
        done = bool(reset_payload.get("done", False))
        rewards: List[float] = []
        steps_taken = 0
        final_reward = 0.0

        while not done and steps_taken < MAX_STEPS:
            steps_taken += 1
            state = observation_to_vector(observation)
            mask = action_mask(observation)
            action_idx, log_prob, value = self.policy.sample_action(state, mask, self.rng)
            action = discrete_to_env_action(action_idx, observation)
            result = self.runner.step_env(session_id, episode_id, action)
            next_observation = result.get("observation", result)
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            episode_id = str(result.get("episode_id", episode_id))
            buffer.add(
                RolloutStep(
                    state=state,
                    action_idx=action_idx,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    mask=mask.astype(np.float64),
                )
            )
            rewards.append(reward)
            final_reward = reward
            observation = next_observation

        return buffer, EpisodeResult(
            session_id=session_id,
            episode_id=episode_id,
            total_reward=float(sum(rewards)),
            final_score=strict_score(final_reward if rewards else 0.0),
            rewards=rewards,
            steps_taken=steps_taken,
        )

    def train(self, episodes: int, task_name: str) -> dict[str, float]:
        aggregate = {
            "episodes": 0.0,
            "avg_return": 0.0,
            "avg_score": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
        total_return = 0.0
        total_score = 0.0

        for episode_idx in range(episodes):
            buffer, result = self.collect_episode(task_name=task_name, seed=episode_idx)
            if len(buffer) == 0:
                continue
            returns, advantages = buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
            states = np.asarray([step.state for step in buffer.steps], dtype=np.float64)
            actions = np.asarray([step.action_idx for step in buffer.steps], dtype=np.int64)
            old_log_probs = np.asarray([step.log_prob for step in buffer.steps], dtype=np.float64)
            masks = np.asarray([step.mask for step in buffer.steps], dtype=np.float64)
            metrics = self.policy.update(
                states=states,
                actions=actions,
                old_log_probs=old_log_probs,
                returns=returns,
                advantages=advantages,
                masks=masks,
                lr=self.learning_rate,
                clip_eps=self.clip_eps,
                entropy_coef=self.entropy_coef,
                value_coef=self.value_coef,
                epochs=self.update_epochs,
            )
            total_return += result.total_reward
            total_score += result.final_score
            aggregate["episodes"] += 1.0
            for key in ("policy_loss", "value_loss", "entropy"):
                aggregate[key] += metrics[key]
            print(
                f"[PPO] episode={episode_idx + 1} steps={result.steps_taken} "
                f"return={result.total_reward:.4f} score={result.final_score:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} value_loss={metrics['value_loss']:.4f}",
                flush=True,
            )

        completed = max(1.0, aggregate["episodes"])
        aggregate["avg_return"] = total_return / completed
        aggregate["avg_score"] = total_score / completed
        aggregate["policy_loss"] /= completed
        aggregate["value_loss"] /= completed
        aggregate["entropy"] /= completed
        return aggregate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run heuristic/model rollouts or train lightweight PPO.")
    parser.add_argument("--mode", choices=["rollout", "ppo-train", "ppo-eval"], default="rollout")
    parser.add_argument("--task", default=TASK_NAME or "binary_decision")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--weights", default=str(PPO_WEIGHTS_PATH))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    runner = EnvironmentRunner(model=model, tokenizer=tokenizer)

    if args.mode == "rollout":
        tasks = [args.task] if args.task else TASKS
        for task_name in tasks:
            runner.run_task(task_name)
        return

    weights_path = Path(args.weights)
    if args.mode == "ppo-train":
        trainer = PPOTrainer(runner)
        metrics = trainer.train(episodes=args.episodes, task_name=args.task)
        trainer.policy.save(weights_path)
        print(json.dumps({"weights": str(weights_path), **metrics}, indent=2), flush=True)
        return

    if not weights_path.exists():
        raise FileNotFoundError(f"PPO weights not found: {weights_path}")
    policy = PPOPolicy.load(weights_path)
    scores = [runner.rollout(args.task, action_source="ppo", ppo_policy=policy, seed=idx, log_steps=False).final_score for idx in range(args.episodes)]
    print(json.dumps({"episodes": args.episodes, "avg_score": float(np.mean(scores)), "weights": str(weights_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
