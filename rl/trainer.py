from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_data
from pipeline.main_pipeline import CreditDecisionPipeline
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

    def train(self, users: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        users = list(users)
        if not users:
            raise ValueError("RL training requires at least one user record.")

        episode_count = min(self.config.episodes, len(users))
        active_users = users[:episode_count]
        all_trajectories: list[Trajectory] = []

        for batch_start in range(0, episode_count, self.config.batch_size):
            batch_users = active_users[batch_start : batch_start + self.config.batch_size]

            trajectories = self.collector.collect(batch_users)
            print(f"[DEBUG] Batch start: {batch_start}, size: {len(trajectories)}")

    # ✅ ADD THIS BLOCK HERE
            actions = [t.summary["decision"] for t in trajectories]

            approve_count = sum(1 for a in actions if str(a).upper() == "APPROVE")
            approve_rate = approve_count / len(actions) if actions else 0.0

    # Apply penalty if too many approvals
            if approve_rate > 0.65:
                penalty=(approve_rate - 0.65) * 0.5 # dynamic penalty scaling
                for t in trajectories:
                    if t.summary["decision"] == "APPROVE":
                        t.total_reward -= penalty   # 🔥 important

    # ----------------------------------

            all_trajectories.extend(trajectories)

            self._update_policy(trajectories)

            rewards = [trajectory.total_reward for trajectory in trajectories]
            print(f"[DEBUG] Total trajectories: {len(all_trajectories)}")
            batch_summary = {
                    "batch_start": batch_start,
                    "batch_size": len(trajectories),
                    "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
                    "min_reward": round(float(np.min(rewards)), 4) if rewards else 0.0,
                    "max_reward": round(float(np.max(rewards)), 4) if rewards else 0.0,
                    "algorithm": self.config.algorithm.lower(),
                }
            self.training_history.append(batch_summary)

            summary = self._build_training_summary(all_trajectories)
        Path(self.config.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.reward_logger.close()
        return summary

    def _build_training_summary(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        rewards = [trajectory.total_reward for trajectory in trajectories]
        decisions = [trajectory.summary["decision"] for trajectory in trajectories]
        return {
            "episodes": len(trajectories),
            "algorithm": self.config.algorithm.lower(),
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
            "approve_rate": round(decisions.count("APPROVE") / len(decisions), 4) if decisions else 0.0,
            "history": self.training_history,
            "reward_log_path": self.config.rewards_path,
        }

    def _update_policy(self, trajectories: list[Trajectory]) -> None:
        algorithm = self.config.algorithm.lower()
        if algorithm in {"ppo", "grpo"} and self._trl_backend_available():
            try:
                self._update_policy_with_trl(trajectories)
                return
            except Exception:
                if self.config.require_trl:
                    raise

        self._update_policy_lightweight(trajectories)

    def _trl_backend_available(self) -> bool:
        try:
            import trl  # noqa: F401
            import unsloth  # noqa: F401
            import datasets  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _update_policy_with_trl(self, trajectories: list[Trajectory]) -> None:
        from datasets import Dataset
        from transformers import AutoTokenizer
        from unsloth import FastLanguageModel

        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise RuntimeError("TRL GRPO backend is unavailable in this environment.") from exc

        records = []
        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            records.append(
                {
                    "prompt": transition.metadata["prompt"],
                    "features": json.dumps(transition.observation["features"]),
                    "reward": float(transition.reward),
                    "preferred_action": transition.action,
                }
            )
        dataset = Dataset.from_list(records)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model_name,
            max_seq_length=512,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name, use_fast=True)

        def reward_fn(prompts, completions, features, reward, preferred_action, **_: Any):
            rewards: list[float] = []
            for completion, reward_value, target_action in zip(completions, reward, preferred_action):
                decision = self.agent2_module.extract_decision(str(completion))
                if decision == str(target_action):
                    rewards.append(float(reward_value))
                else:
                    rewards.append(max(float(reward_value) - 1.0, -1.0))
            return rewards

        training_args = GRPOConfig(
            output_dir=str(ROOT / "agent2-decision-base" / "checkpoints" / "rl"),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=max(1, min(self.config.batch_size, 4)),
            num_generations=2,
            max_prompt_length=384,
            max_completion_length=4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=reward_fn,
        )
        trainer.train()

    def _update_policy_lightweight(self, trajectories: list[Trajectory]) -> None:
        rewards = np.array([trajectory.total_reward for trajectory in trajectories], dtype=float)
        baseline = float(np.mean(rewards)) if len(rewards) else 0.0
        update_samples: list[dict[str, Any]] = []

        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            action = transition.action
            reward = float(transition.reward)
            advantage = reward - baseline
            if advantage >= 0:
                target_label = action
            else:
                target_label = "REJECT" if action == "APPROVE" else "APPROVE"

            update_samples.append(
                {
                    "features": transition.observation["features"],
                    "risk_score": transition.observation["risk_score"],
                    "shap_info": transition.observation["shap_info"],
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
