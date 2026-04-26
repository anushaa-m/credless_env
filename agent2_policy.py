import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PolicyParams:
    """All learnable parameters in one place. Serialisable to JSON."""

    weight: float = -3.0
    bias: float = 1.5
    lr: float = 0.3
    baseline: float = 0.0
    baseline_alpha: float = 0.05


class Agent2Policy:
    """
    Probabilistic decision policy for Agent2.

    Decision: p(approve) = sigmoid(weight * risk_score + bias)
    Sampling: action ~ Bernoulli(p_approve)
    Update: REINFORCE gradient ascent on expected reward.
    """

    def __init__(self, params: Optional[PolicyParams] = None, checkpoint_path: str = "policy_checkpoint.json"):
        self.p = params or PolicyParams()
        self.checkpoint_path = checkpoint_path
        self._load_if_exists()

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def p_approve(self, risk_score: float) -> float:
        """Probability of approving given a risk score in [0, 1]."""
        return self._sigmoid(self.p.weight * risk_score + self.p.bias)

    def sample_action(self, risk_score: float) -> tuple[str, float]:
        """
        Sample a decision and return it with its log-probability.

        Returns:
            action: "approve" or "deny"
            log_prob: log pi(action | risk_score), needed for REINFORCE update.
        """
        import random

        pa = self.p_approve(risk_score)
        action = "approve" if random.random() < pa else "deny"
        log_prob = math.log(pa + 1e-9) if action == "approve" else math.log(1 - pa + 1e-9)
        return action, log_prob

    def update(self, risk_score: float, action: str, reward: float) -> dict:
        """
        One REINFORCE gradient step.

        d/dw log pi(approve) = (1 - p_approve) * risk_score
        d/dw log pi(deny) = -p_approve * risk_score
        Same pattern for bias with risk_score replaced by 1.
        """
        pa = self.p_approve(risk_score)
        advantage = reward - self.p.baseline

        normalized_action = str(action).strip().lower()
        if normalized_action == "reject":
            normalized_action = "deny"

        if normalized_action == "approve":
            grad_w = (1.0 - pa) * risk_score
            grad_b = 1.0 - pa
        else:
            grad_w = -pa * risk_score
            grad_b = -pa

        self.p.weight += self.p.lr * advantage * grad_w
        self.p.bias += self.p.lr * advantage * grad_b
        self.p.weight = max(-10.0, min(10.0, self.p.weight))
        self.p.bias = max(-5.0, min(5.0, self.p.bias))
        self.p.baseline += self.p.baseline_alpha * (reward - self.p.baseline)

        return {
            "weight": round(self.p.weight, 4),
            "bias": round(self.p.bias, 4),
            "p_approve": round(pa, 4),
            "advantage": round(advantage, 4),
        }

    def save(self):
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.p), f, indent=2)

    def _load_if_exists(self):
        if Path(self.checkpoint_path).exists():
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(self.p, k):
                    setattr(self.p, k, v)
            print(f"[Agent2Policy] Loaded checkpoint: w={self.p.weight:.3f} b={self.p.bias:.3f}")
