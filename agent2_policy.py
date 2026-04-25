from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PolicyParams:
    weight: float = -3.0
    bias: float = 1.5
    lr: float = 0.3
    baseline: float = 0.0
    baseline_alpha: float = 0.05


class Agent2Policy:
    def __init__(
        self,
        params: PolicyParams | None = None,
        checkpoint_path: str | Path = "policy_checkpoint.json",
    ) -> None:
        self.p = params or PolicyParams()
        self.checkpoint_path = Path(checkpoint_path)
        self._load_if_exists()

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def p_approve(self, risk_score: float) -> float:
        return self._sigmoid(self.p.weight * float(risk_score) + self.p.bias)

    def sample_action(self, risk_score: float) -> tuple[str, float]:
        approve_probability = self.p_approve(risk_score)
        action = "APPROVE" if random.random() < approve_probability else "REJECT"
        if action == "APPROVE":
            log_prob = math.log(approve_probability + 1e-9)
        else:
            log_prob = math.log(1.0 - approve_probability + 1e-9)
        return action, float(log_prob)

    def update(self, risk_score: float, action: str, reward: float) -> dict[str, float]:
        risk_score = float(risk_score)
        reward = float(reward)
        approve_probability = self.p_approve(risk_score)
        advantage = reward - self.p.baseline

        if str(action).strip().upper() == "APPROVE":
            grad_w = (1.0 - approve_probability) * risk_score
            grad_b = 1.0 - approve_probability
        else:
            grad_w = -approve_probability * risk_score
            grad_b = -approve_probability

        self.p.weight += self.p.lr * advantage * grad_w
        self.p.bias += self.p.lr * advantage * grad_b
        self.p.weight = max(-10.0, min(10.0, self.p.weight))
        self.p.bias = max(-5.0, min(5.0, self.p.bias))
        self.p.baseline += self.p.baseline_alpha * (reward - self.p.baseline)

        return {
            "weight": round(float(self.p.weight), 4),
            "bias": round(float(self.p.bias), 4),
            "p_approve": round(float(approve_probability), 4),
            "advantage": round(float(advantage), 4),
        }

    def save(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(json.dumps(asdict(self.p), indent=2), encoding="utf-8")

    def _load_if_exists(self) -> None:
        if not self.checkpoint_path.exists():
            return

        data: dict[str, Any] = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(self.p, key):
                setattr(self.p, key, float(value))
