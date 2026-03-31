# server/environment.py
from openenv.core.env_server.interfaces import Environment   # ✅ correct path
from models import CreditAction, CreditObservation, CreditState
from .oracle import CredLessOracle
import uuid
from typing import Optional

from server.data_generator import (
    generate_applicant, FIELD_RANGES,
    ALWAYS_VISIBLE, HIDDEN_INITIALLY,
)
from server.graders import (
    grade_binary_decision,
    grade_risk_tiering,
    grade_adaptive_inquiry,
)

TASKS     = ["binary_decision", "risk_tiering", "adaptive_inquiry"]
MAX_STEPS = 12


class CreditAnalystEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self.oracle          = CredLessOracle()
        self._applicant      = {}
        self._ground_truth   = {}       # ← populated by reset(), read by step()
        self._task           = "binary_decision"
        self._steps          = 0
        self._requests       = []
        self._cum_reward     = 0.0
        self._episode_id     = ""

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(
        self,
        task_name: str = "binary_decision",
        seed: Optional[int] = None,
    ) -> CreditObservation:

        # Sanitise task name
        if task_name not in TASKS:
            task_name = "binary_decision"

        # Generate fresh applicant + ground truth
        self._applicant    = generate_applicant(seed)
        self._ground_truth = self.oracle.predict(self._applicant["features"])

        # ✅ FIX: use str(uuid.uuid4()) — NOT the literal `...`
        self._episode_id  = str(uuid.uuid4())
        self._task        = task_name
        self._steps       = 0
        self._requests    = []
        self._cum_reward  = 0.0

        # Guard: confirm oracle returned expected keys
        assert "decision" in self._ground_truth, (
            f"Oracle missing 'decision' key. Got: {self._ground_truth}"
        )
        assert "tier"     in self._ground_truth, (
            f"Oracle missing 'tier' key. Got: {self._ground_truth}"
        )

        # Reveal fields based on task
        if task_name == "adaptive_inquiry":
            revealed = {k: self._applicant["features"][k] for k in ALWAYS_VISIBLE}
            hidden   = list(HIDDEN_INITIALLY)
        else:
            revealed = dict(self._applicant["features"])
            hidden   = []

        return CreditObservation(
            applicant_id    = self._applicant["applicant_id"],
            revealed_fields = revealed,
            hidden_fields   = hidden,
            task_name       = task_name,
            message         = (
                f"New applicant '{self._applicant['applicant_id']}'. "
                f"Task: '{task_name}'. Analyse and take action."
            ),
        )

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: CreditAction) -> CreditObservation:
        # Guard: step called before reset
        if not self._ground_truth:
            return CreditObservation(
                task_name = self._task,
                done      = True,
                message   = "ERROR: call /reset before /step.",
                step_reward = -1.0,
            )

        self._steps += 1
        reward        = 0.0
        done          = False
        episode_score = 0.0
        message       = ""

        atype = (action.action_type or "").strip().lower()
        task  = self._task

        # ── Gather information ────────────────────────────────────────────────
        if atype == "request_field":
            fname = (action.field_name or "").strip()

            if task != "adaptive_inquiry":
                reward  = -0.05
                message = "Field requests only valid in 'adaptive_inquiry'."

            elif fname not in FIELD_RANGES:
                reward  = -0.05
                message = f"Unknown field '{fname}'. Valid: {list(FIELD_RANGES.keys())}"

            elif fname in self._requests:
                reward  = -0.10
                message = f"Duplicate request '{fname}' — penalised."

            else:
                self._requests.append(fname)
                val     = self._applicant["features"][fname]
                reward  = 0.05
                message = f"Revealed: '{fname}' = {val:.4f}"

        # ── Final decision ────────────────────────────────────────────────────
        elif atype in ("approve", "deny", "assign_tier"):
            done = True

            if task == "binary_decision":
                dec           = (action.decision or "deny").strip().lower()
                episode_score = grade_binary_decision(
                    dec, self._ground_truth["decision"]
                )
                reward  = episode_score
                message = (
                    f"Decision: '{dec}' | "
                    f"Oracle: '{self._ground_truth['decision']}' | "
                    f"Score: {episode_score:.2f}"
                )

            elif task == "risk_tiering":
                tier          = (action.tier or "medium_risk").strip().lower()
                limit         = float(action.credit_limit or 0.0)
                episode_score = grade_risk_tiering(
                    tier, self._ground_truth["tier"],
                    limit, self._ground_truth["default_prob"],
                )
                reward  = episode_score
                message = (
                    f"Tier: '{tier}' | Oracle: '{self._ground_truth['tier']}' | "
                    f"Limit: {limit:,.0f} INR | Score: {episode_score:.2f}"
                )

            elif task == "adaptive_inquiry":
                dec           = (action.decision or "deny").strip().lower()
                episode_score = grade_adaptive_inquiry(
                    dec, self._ground_truth["decision"],
                    len(self._requests),
                )
                reward  = episode_score
                message = (
                    f"Decision: '{dec}' | "
                    f"Fields used: {len(self._requests)} | "
                    f"Score: {episode_score:.2f}"
                )

        else:
            reward  = -0.05
            message = f"Unknown action_type '{atype}'."

        # ── Timeout guard ─────────────────────────────────────────────────────
        if self._steps >= MAX_STEPS and not done:
            done          = True
            reward        = -0.5
            episode_score = 0.0
            message       = "Episode timed out."

        self._cum_reward += reward

        # Build current field view
        if self._task == "adaptive_inquiry":
            visible  = ALWAYS_VISIBLE + self._requests
            revealed = {k: self._applicant["features"][k]
                        for k in visible if k in self._applicant["features"]}
            hidden   = [f for f in FIELD_RANGES if f not in revealed]
        else:
            revealed = dict(self._applicant["features"])
            hidden   = []

        return CreditObservation(
            applicant_id      = self._applicant.get("applicant_id", ""),
            revealed_fields   = revealed,
            hidden_fields     = hidden,
            task_name         = self._task,
            step_reward       = round(reward, 4),
            cumulative_reward = round(self._cum_reward, 4),
            done              = done,
            message           = message,
            episode_score     = round(episode_score, 4),
        )

    # ── state property ────────────────────────────────────────────────────────
    @property
    def state(self) -> CreditState:
        return CreditState(
        episode_id            = self._episode_id,
        task_name             = self._task,
        step_count            = self._steps,
        cumulative_reward     = round(self._cum_reward, 4),
        fields_requested      = list(self._requests),
        ground_truth_tier     = self._ground_truth.get("tier", ""),
        ground_truth_decision = self._ground_truth.get("decision", ""),
        ground_truth_prob     = self._ground_truth.get("default_prob", 0.0),
    )