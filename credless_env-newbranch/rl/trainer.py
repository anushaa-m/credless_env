from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows stability: TRL bundles UTF-8 jinja templates; Python's default cp1252 can
# break imports. If we are not in UTF-8 mode, re-run once with `-X utf8`.
if os.name == "nt" and not sys.flags.utf8_mode and os.environ.get("CREDLESS_UTF8_REEXEC") != "1":
    os.environ["CREDLESS_UTF8_REEXEC"] = "1"
    cmd = [sys.executable, "-X", "utf8", "-m", "rl.trainer", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_data
from pipeline.main_pipeline import CreditDecisionPipeline
from pipeline.oracle import oracle_decision
from agent2_llm import Agent2LLMPolicy
from rl.diagnostics import write_reward_curve
from rl.reward_logger import RewardLogger
from rl.rollout_collector import RolloutCollector, Trajectory


@dataclass
class RLTrainingConfig:
    algorithm: str = "grpo"
    episodes: int = 256
    batch_size: int = 32
    seed: int = 42
    rewards_path: str = "rl/reward_log.jsonl"
    summary_path: str = "rl/training_summary.json"
    require_trl: bool = False
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    learning_rate: float = 1e-5
    output_dir: str = "agent2_llm_checkpoints/grpo"
    approve_rate_min: float = 0.30
    approve_rate_max: float = 0.70
    approve_rate_penalty: float = 0.35


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
        # Ensure rollouts are produced by the same LLM base used for GRPO fine-tuning.
        self.pipeline.agent2 = Agent2LLMPolicy(model_name_or_path=self.config.base_model_name)
        self.reward_logger = reward_logger or RewardLogger(self.config.rewards_path, flush_every=1)
        self.collector = RolloutCollector(self.pipeline)
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

            self._apply_approve_rate_penalty_inplace(trajectories)
            self._log_trajectories(trajectories)

            all_trajectories.extend(trajectories)

            self._update_policy(trajectories)

            rewards = [trajectory.total_reward for trajectory in trajectories]
            decisions = [str(t.summary.get("decision", "")).upper() for t in trajectories]
            approve_rate = decisions.count("APPROVE") / max(1, len(decisions))
            print(f"[DEBUG] Total trajectories: {len(all_trajectories)}")
            batch_summary = {
                    "batch_start": batch_start,
                    "batch_size": len(trajectories),
                    "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
                    "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
                    "min_reward": round(float(np.min(rewards)), 4) if rewards else 0.0,
                    "max_reward": round(float(np.max(rewards)), 4) if rewards else 0.0,
                    "approve_rate": round(float(approve_rate), 4),
                    "decision_distribution": {
                        "APPROVE": int(decisions.count("APPROVE")),
                        "REJECT": int(decisions.count("REJECT")),
                    },
                    "algorithm": self.config.algorithm.lower(),
                }
            self.training_history.append(batch_summary)

            summary = self._build_training_summary(all_trajectories)
        Path(self.config.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifacts = write_reward_curve(self.training_history)
        if artifacts:
            summary["artifacts"] = artifacts
            Path(self.config.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.reward_logger.close()
        return summary

    def _log_trajectories(self, trajectories: list[Trajectory]) -> None:
        for trajectory in trajectories:
            if not trajectory.transitions:
                continue
            self.reward_logger.log_step(asdict(trajectory.transitions[0]))
            self.reward_logger.log_episode(dict(trajectory.summary))

    def _apply_approve_rate_penalty_inplace(self, trajectories: list[Trajectory]) -> None:
        if not trajectories:
            return
        decisions = [str(t.summary.get("decision", "")).upper() for t in trajectories]
        approve_rate = decisions.count("APPROVE") / max(1, len(decisions))

        # Keep reward shaping simple and visible: discourage collapse to all-approve/all-reject.
        penalty = 0.0
        if approve_rate < self.config.approve_rate_min:
            penalty = (self.config.approve_rate_min - approve_rate) * self.config.approve_rate_penalty
        elif approve_rate > self.config.approve_rate_max:
            penalty = (approve_rate - self.config.approve_rate_max) * self.config.approve_rate_penalty

        if penalty <= 0.0:
            return

        for t in trajectories:
            decision = str(t.summary.get("decision", "")).upper()
            if decision == "APPROVE" and approve_rate > self.config.approve_rate_max:
                t.total_reward = float(t.total_reward) - float(penalty)
                t.summary["total_reward"] = float(t.total_reward)
                if t.transitions:
                    t.transitions[0].reward = float(t.total_reward)
            if decision == "REJECT" and approve_rate < self.config.approve_rate_min:
                t.total_reward = float(t.total_reward) - float(penalty)
                t.summary["total_reward"] = float(t.total_reward)
                if t.transitions:
                    t.transitions[0].reward = float(t.total_reward)

    def _build_training_summary(self, trajectories: list[Trajectory]) -> dict[str, Any]:
        rewards = [trajectory.total_reward for trajectory in trajectories]
        decisions = [trajectory.summary["decision"] for trajectory in trajectories]
        decisions_upper = [str(d).upper() for d in decisions]
        return {
            "episodes": len(trajectories),
            "algorithm": self.config.algorithm.lower(),
            "mean_reward": round(float(np.mean(rewards)), 4) if rewards else 0.0,
            "std_reward": round(float(np.std(rewards)), 4) if rewards else 0.0,
            "approve_rate": round(decisions_upper.count("APPROVE") / len(decisions_upper), 4) if decisions_upper else 0.0,
            "decision_distribution": {
                "APPROVE": int(decisions_upper.count("APPROVE")),
                "REJECT": int(decisions_upper.count("REJECT")),
            },
            "history": self.training_history,
            "reward_log_path": self.config.rewards_path,
        }

    def _update_policy(self, trajectories: list[Trajectory]) -> None:
        algorithm = self.config.algorithm.lower()
        if algorithm != "grpo":
            raise ValueError(f"Unsupported algorithm '{self.config.algorithm}'. Use --algorithm grpo.")
        if not self._trl_backend_available():
            raise RuntimeError("TRL+Unsloth backend not available. Install `trl`, `unsloth`, `datasets`, `transformers`.")
        self._update_policy_with_trl(trajectories)

    def _trl_backend_available(self) -> bool:
        try:
            import trl  # noqa: F401
            import datasets  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except ImportError:
            return False
        return True

    def _update_policy_with_trl(self, trajectories: list[Trajectory]) -> None:
        from datasets import Dataset
        from transformers import AutoTokenizer

        try:
            from trl import GRPOConfig, GRPOTrainer
        except ImportError as exc:
            raise RuntimeError("TRL GRPO backend is unavailable in this environment.") from exc

        records = []
        for trajectory in trajectories:
            transition = trajectory.transitions[0]
            features = dict(transition.observation.get("features", {}))
            oracle = oracle_decision(features)
            records.append(
                {
                    "prompt": transition.metadata["prompt"],
                    "oracle_decision": "APPROVE" if oracle["decision"] == "approve" else "REJECT",
                    "oracle_confidence": float(oracle["confidence"]),
                    "efficiency_penalty": 0.01,
                }
            )
        dataset = Dataset.from_list(records)

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name, use_fast=False)

        # Prefer Unsloth when available; otherwise fall back to pure Transformers + PEFT LoRA.
        model = None
        try:
            from unsloth import FastLanguageModel  # type: ignore

            model, _ = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model_name,
                max_seq_length=512,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        except Exception:
            from peft import LoraConfig, get_peft_model  # type: ignore
            from transformers import AutoModelForCausalLM  # type: ignore

            base = AutoModelForCausalLM.from_pretrained(self.config.base_model_name)
            model_type = str(getattr(getattr(base, "config", None), "model_type", "") or "").lower()
            if model_type in {"gpt2"}:
                target_modules = ["c_attn", "c_proj"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(base, lora_cfg)
        assert model is not None

        from agent2_llm import extract_decision

        def reward_fn(prompts, completions, oracle_decision, oracle_confidence, efficiency_penalty, **_: Any):
            raw_rewards: list[float] = []
            decisions: list[str] = []
            for completion, oracle_label, confidence, penalty in zip(
                completions,
                oracle_decision,
                oracle_confidence,
                efficiency_penalty,
            ):
                decision = extract_decision(str(completion))
                if decision == "INVALID":
                    decision = "REJECT"
                decisions.append(decision)
                match = 1.0 if decision == str(oracle_label).upper() else -1.0
                # Primary signal: oracle alignment, weighted by oracle confidence.
                reward = match * (0.5 + float(confidence))
                # Small, always-on efficiency penalty (encourages single-shot output).
                reward -= float(penalty)
                raw_rewards.append(float(reward))

            # Approve-rate shaping (must affect training): discourage collapse to all-approve/all-reject.
            approve_rate = decisions.count("APPROVE") / max(1, len(decisions))
            if approve_rate > self.config.approve_rate_max:
                rate_penalty = (approve_rate - self.config.approve_rate_max) * self.config.approve_rate_penalty
                raw_rewards = [
                    float(r - rate_penalty) if d == "APPROVE" else float(r)
                    for r, d in zip(raw_rewards, decisions)
                ]
            elif approve_rate < self.config.approve_rate_min:
                rate_penalty = (self.config.approve_rate_min - approve_rate) * self.config.approve_rate_penalty
                raw_rewards = [
                    float(r - rate_penalty) if d == "REJECT" else float(r)
                    for r, d in zip(raw_rewards, decisions)
                ]

            # Batch advantage baseline (stability): reward - mean(batch_reward)
            mean_r = float(np.mean(raw_rewards)) if raw_rewards else 0.0
            return [float(r - mean_r) for r in raw_rewards]

        training_args = GRPOConfig(
            output_dir=str(Path(self.config.output_dir)),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=max(1, min(self.config.batch_size, 4)),
            num_generations=2,
            max_completion_length=6,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            use_cpu=True,
            bf16=False,
            fp16=False,
        )
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )
        trainer.train()
        # Persist the fine-tuned adapter/model for inference.
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))


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
    parser.add_argument("--algorithm", type=str, default="grpo", choices=["grpo"])
    parser.add_argument("--require-trl", action="store_true")
    parser.add_argument("--base-model", type=str, default=RLTrainingConfig.base_model_name)
    parser.add_argument("--learning-rate", type=float, default=RLTrainingConfig.learning_rate)
    parser.add_argument("--output-dir", type=str, default=RLTrainingConfig.output_dir)
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
        base_model_name=args.base_model,
        learning_rate=float(args.learning_rate),
        output_dir=str(args.output_dir),
    )
    trainer = RLTrainer(config=config)
    summary = trainer.train(users)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
