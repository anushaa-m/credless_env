from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_data
from pipeline.main_pipeline import CreditDecisionEnvironment, CreditDecisionPipeline
from rl.reward_logger import RewardLogger
from rl.rollout_collector import RolloutCollector, Trajectory


def _load_agent2_module():
    module_path = ROOT / "agent2-decision-base" / "train.py"
    module_name = "credless_agent2_train"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Agent 2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class RLTrainingConfig:
    algorithm: str = "ppo"
    episodes: int = 256
    batch_size: int = 32
    seed: int = 42
    rewards_path: str = "rl/reward_log.jsonl"
    summary_path: str = "rl/training_summary.json"
    require_trl: bool = False
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 1e-5
    max_seq_length: int = 512
    ppo_batch_size: int = 4
    ppo_mini_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    lora_rank: int = 32
    lora_alpha: int = 32
    max_new_tokens: int = 4


class RLTrainer:
    def __init__(
        self,
        config: RLTrainingConfig | None = None,
        *,
        pipeline: CreditDecisionPipeline | None = None,
        reward_logger: RewardLogger | None = None,
    ) -> None:
        self.config = config or RLTrainingConfig()
        self.pipeline = pipeline or CreditDecisionPipeline()
        self.reward_logger = reward_logger or RewardLogger(self.config.rewards_path, flush_every=1)
        self.collector = RolloutCollector(self.pipeline, reward_logger=self.reward_logger)
        self.agent2_module = _load_agent2_module()
        self.training_history: list[dict[str, Any]] = []
        self._trl_runtime: dict[str, Any] | None = None
     
    def build_prompt(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: list[Mapping[str, Any]] | None = None,
    ) -> str:

        features = dict(features or {})

        # 🔥 PRIORITIZE IMPORTANT FEATURES
        priority_keys = [
            "age", "monthlyincome", "debtratio",
            "numberofopencreditlinesandloans",
            "numberoftimes90dayslate",
            "avg_monthly_balance",
            "overdraft_count",
            "salary_credit_consistency",
            "income_variability_score",
        ]

        priority_features = []
        other_features = []

        for k, v in features.items():
            text = f"{k}: {round(float(v), 3)}" if isinstance(v, (int, float)) else f"{k}: {v}"
            if k in priority_keys:
                priority_features.append(text)
            else:
                other_features.append(text)

        # limit size (IMPORTANT for PPO)
        other_features = other_features[:15]

        feature_text = "\n".join(priority_features + other_features)

        # 🔥 SHAP (top 3 only)
        shap_text = "\n".join([
            f"{item['feature']} ({item['impact']})"
            for item in (shap_info or [])[:3]
            if "feature" in item and "impact" in item
        ])

        if not shap_text:
            shap_text = "None"

        return f"""
Credit Risk Decision Task

Risk Score: {risk_score:.3f}

Key Features:
{feature_text}

Top Signals:
{shap_text}

Instruction:
- Approve financially stable users
- Reject high-risk or unstable users
- Use BOTH risk score and features

Answer ONLY:
APPROVE or REJECT
"""


    def _build_ppo_records(self, trajectories: list[Trajectory]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            records.append(
                {
                    "query": self.build_prompt(
                        transition.observation.get("features", {}),
                        transition.observation.get("risk_score", 0.5),
                        transition.observation.get("shap_info", []),
                    ),
                    "response": "",
                    "reward": float(transition.reward),
                    "oracle_decision": str(trajectory.summary.get("oracle_decision", "REJECT")),
                }
            )
        return records

    def train(self, users: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        users = list(users)
        if not users:
            raise ValueError("RL training requires at least one user record.")

        episode_count = min(self.config.episodes, len(users))
        active_users = users[:episode_count]
        all_trajectories: list[Trajectory] = []
        trajectory_records: list[dict[str, Any]] = []

        for batch_start in range(0, episode_count, self.config.batch_size):
            batch_users = active_users[batch_start : batch_start + self.config.batch_size]
            trajectories = self.collector.collect(batch_users)
            sanitized_trajectories = self._sanitize_trajectories(trajectories)

            actions = [trajectory.summary["decision"] for trajectory in sanitized_trajectories]
            approve_rate = (
                sum(1 for action in actions if action == "APPROVE") / len(actions)
                if actions
                else 0.0
            )

            if approve_rate > 0.65:
                penalty = (approve_rate - 0.65) * 0.5
                for trajectory in sanitized_trajectories:
                    if trajectory.summary["decision"] == "APPROVE":
                        trajectory.total_reward -= penalty
                        trajectory.transitions[0].reward = float(trajectory.total_reward)
                        trajectory.summary["total_reward"] = float(trajectory.total_reward)

            all_trajectories.extend(sanitized_trajectories)
            self._update_policy(sanitized_trajectories)

            rewards = [trajectory.total_reward for trajectory in sanitized_trajectories]
            agreements = [int(trajectory.summary["agreement"]) for trajectory in sanitized_trajectories]
            for trajectory in sanitized_trajectories:
                decision = str(trajectory.summary["decision"])
                reward = float(trajectory.total_reward)
                oracle_decision = str(trajectory.summary["oracle_decision"])
                agreement = int(trajectory.summary["agreement"])
                print(f"[DEBUG] Action: {decision}, Reward: {reward}")
                print(f"[DEBUG] Oracle Decision: {oracle_decision}, Agreement: {agreement}")
                trajectory_records.append({"action": decision, "reward": reward})

            self.training_history.append(
                {
                    "batch_start": batch_start,
                    "batch_size": len(sanitized_trajectories),
                    "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
                    "min_reward": round(float(np.min(rewards)), 4) if rewards else 0.0,
                    "max_reward": round(float(np.max(rewards)), 4) if rewards else 0.0,
                    "approve_rate": round(float(approve_rate), 4),
                    "oracle_agreement_rate": round(float(np.mean(agreements)), 4) if agreements else 0.0,
                    "algorithm": self.config.algorithm.lower(),
                }
            )

        summary = self._build_training_summary(all_trajectories)
        summary["trajectories"] = trajectory_records
        Path(self.config.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.reward_logger.close()
        return summary

    def _sanitize_trajectories(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        sanitized: list[Trajectory] = []
        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            model_output = str(transition.metadata.get("raw_text", trajectory.summary.get("decision", "")))
            decision = self.agent2_module.extract_decision(model_output)
            if decision == "INVALID":
                decision = self.agent2_module.extract_decision(str(trajectory.summary.get("decision", "")))
            if decision == "INVALID":
                decision = "REJECT"

            reward = float(transition.reward if transition.reward is not None else trajectory.total_reward)
            info = dict(transition.info or {})
            oracle_decision = str(info.get("oracle_decision", "REJECT")).upper()
            agreement = int(decision == oracle_decision)

            transition.action = decision
            transition.reward = reward
            transition.done = True
            transition.metadata["prompt"] = self.build_prompt(
                transition.observation.get("features", {}),
                float(transition.observation.get("risk_score", 0.5)),
                transition.observation.get("shap_info", []) or [],
            )
            trajectory.total_reward = reward
            trajectory.done = True
            trajectory.summary["decision"] = decision
            trajectory.summary["done"] = True
            trajectory.summary["total_reward"] = reward
            trajectory.summary["oracle_decision"] = oracle_decision
            trajectory.summary["agreement"] = agreement
            trajectory.summary["oracle_score"] = float(info.get("oracle_score", reward))
            sanitized.append(trajectory)
        return sanitized

    def _build_training_summary(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        rewards = [trajectory.total_reward for trajectory in trajectories]
        decisions = [trajectory.summary["decision"] for trajectory in trajectories]
        agreements = [int(trajectory.summary.get("agreement", 0)) for trajectory in trajectories]
        return {
            "episodes": len(trajectories),
            "algorithm": self.config.algorithm.lower(),
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
            "approve_rate": round(decisions.count("APPROVE") / len(decisions), 4) if decisions else 0.0,
            "oracle_agreement_rate": round(float(np.mean(agreements)), 4) if agreements else 0.0,
            "history": self.training_history,
            "reward_log_path": self.config.rewards_path,
        }

    def _update_policy(self, trajectories: list[Trajectory]) -> None:
        algorithm = self.config.algorithm.lower()
        if algorithm == "ppo" and self._trl_backend_available():
            try:
                self._update_policy_with_trl(trajectories)
                return
            except Exception:
                if self.config.require_trl:
                    raise
        self._update_policy_lightweight(trajectories)

    def _trl_backend_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import trl  # noqa: F401
            import unsloth  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_trl_runtime(self) -> dict[str, Any]:
        if self._trl_runtime is not None:
            return self._trl_runtime

        import torch
        from transformers import AutoTokenizer
        from trl import PPOConfig, PPOTrainer
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.ppo_batch_size,
            mini_batch_size=self.config.ppo_mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        trainer = PPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=ppo_config,
        )

        self._trl_runtime = {
            "torch": torch,
            "trainer": trainer,
            "tokenizer": tokenizer,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        return self._trl_runtime

    def _update_policy_with_trl(self, trajectories: list[Trajectory]) -> None:
        runtime = self._get_trl_runtime()
        torch = runtime["torch"]
        trainer = runtime["trainer"]
        tokenizer = runtime["tokenizer"]
        device = runtime["device"]

        # 🔥 Batch buffers
        queries = []
        responses = []
        rewards = []
        batch_size = 32

        # 🔥 Metrics tracking
        total_rewards_log = []
        agreements_log = []
        decisions_log = []

        records = self._build_ppo_records(trajectories)

        for trajectory, record in zip(trajectories, records):
            transition = trajectory.transitions[0]

            features_dict = dict(transition.observation.get("features", {}))
            prompt = record["query"]

            # 🔹 Tokenize
            query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            # 🔹 Generate response
            generated = trainer.generate(
                query_tensor,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )

            response_tensor = generated[:, query_tensor.shape[-1]:]
            response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

            # 🔹 Extract decision
            decision = self.agent2_module.extract_decision(response_text)
            if decision == "INVALID":
                decision = "REJECT"

            # 🔹 Environment step
            result = self.pipeline.run(features_dict)
            if not isinstance(result, dict):
                raise ValueError("Expected environment result to be a dict for PPO training.")

            info = dict(result.get("info", {}))
            oracle_score = float(info.get("oracle_score", 0.0))
            oracle_decision = str(info.get("oracle_decision", "REJECT")).upper()

            # 🔥 Hybrid reward
            reward = (1.0 if decision == oracle_decision else -1.0) + 0.5 * oracle_score

            # 🔥 Metrics logging
            total_rewards_log.append(reward)
            agreements_log.append(int(decision == oracle_decision))
            decisions_log.append(decision)

            # 🔥 Collect batch
            queries.append(query_tensor[0])
            responses.append(response_tensor[0])
            rewards.append(float(reward))

            # 🔥 Batch update
            if len(queries) >= batch_size:
                reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)

                # normalize rewards
                reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

                trainer.step(queries, responses, reward_tensor)

                print(f"[PPO] Batch Reward Mean: {reward_tensor.mean().item():.4f}")
                print(f"[PPO] Batch Reward Std: {reward_tensor.std().item():.4f}")

                # reset batch
                queries, responses, rewards = [], [], []

            # 🔹 Update trajectory
            transition.action = decision
            transition.reward = float(reward)
            transition.done = True
            transition.info = info

            trajectory.total_reward += float(reward)
            trajectory.done = True
            trajectory.summary["decision"] = decision
            trajectory.summary["total_reward"] = float(trajectory.total_reward)
            trajectory.summary["oracle_decision"] = oracle_decision
            trajectory.summary["oracle_score"] = oracle_score
            trajectory.summary["agreement"] = int(decision == oracle_decision)

            # 🔹 Debug logs
            print(f"[DEBUG] Action: {decision}, Reward: {reward}")
            print(f"[DEBUG] Oracle Decision: {oracle_decision}, Agreement: {int(decision == oracle_decision)}")

        # 🔥 FINAL leftover batch
        if len(queries) > 0:
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-8)

            trainer.step(queries, responses, reward_tensor)

            print(f"[PPO] Final Batch Reward Mean: {reward_tensor.mean().item():.4f}")

        # 🔥 FINAL TRAINING SUMMARY
        import numpy as np

        if total_rewards_log:
            avg_reward = np.mean(total_rewards_log)
            agreement_rate = np.mean(agreements_log)
            approve_rate = decisions_log.count("APPROVE") / len(decisions_log)

            print("\n===== TRAINING SUMMARY =====")
            print(f"Avg Reward: {avg_reward:.4f}")
            print(f"Oracle Agreement: {agreement_rate:.4f}")
            print(f"Approve Rate: {approve_rate:.4f}")
           
    def _update_policy_lightweight(self, trajectories: list[Trajectory]) -> None:
        rewards = np.array([trajectory.total_reward for trajectory in trajectories], dtype=float)
        baseline = float(np.mean(rewards)) if len(rewards) else 0.0
        update_samples: list[dict[str, Any]] = []

        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            action = str(transition.action)
            reward = float(transition.reward)
            advantage = reward - baseline
            target_label = action if advantage >= 0 else ("REJECT" if action == "APPROVE" else "APPROVE")
            update_samples.append(
                {
                    "features": transition.observation.get("features", {}),
                    "risk_score": float(transition.observation.get("risk_score", 0.5)),
                    "shap_info": transition.observation.get("shap_info") or [],
                    "label": target_label,
                    "weight": 1.0 + abs(advantage),
                }
            )

        self.pipeline.agent2.partial_fit_from_feedback(update_samples)
        self.pipeline.agent2.save()


def _load_users(csv_path: str | None, n_rows: int, seed: int) -> list[dict[str, Any]]:
    if csv_path:
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        if n_rows > 0 and len(frame) > n_rows:
            frame = frame.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    else:
        frame = generate_synthetic_data(n_samples=n_rows, seed=seed, include_target=False)
    return frame.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL fine-tuning loop for Agent 2.")
    parser.add_argument("--csv", type=str, default=None, help="Optional dataset CSV.")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "grpo"])
    parser.add_argument("--require-trl", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    users = _load_users(args.csv, args.episodes, args.seed)
    config = RLTrainingConfig(
        algorithm=args.algorithm,
        episodes=args.episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        require_trl=args.require_trl,
    )
    trainer = RLTrainer(config=config)
    summary = trainer.train(users)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
