from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DECISION_RE = re.compile(r"\b(APPROVE|REJECT)\b", re.IGNORECASE)


def extract_decision(text: str) -> str:
    match = DECISION_RE.search(text or "")
    if not match:
        return "INVALID"
    return match.group(1).upper()


def format_prompt(
    features: Mapping[str, Any],
    risk_score: float,
    shap_info: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    top_factors = []
    for item in list(shap_info or [])[:6]:
        name = str(item.get("feature", "") or item.get("name", "") or "").strip()
        contrib = float(item.get("contribution", 0.0) or 0.0)
        if name:
            top_factors.append(f"- {name}: {contrib:+.4f}")

    factors_block = "\n".join(top_factors) if top_factors else "- (none)"

    # Deterministic output contract for GRPO.
    return (
        "You are Agent2, a cautious credit decision policy.\n"
        "Your job: output a single-line final decision.\n\n"
        "Context:\n"
        f"- risk_score (0=low risk, 1=high risk): {float(risk_score):.4f}\n"
        f"- top_factors (SHAP-like):\n{factors_block}\n\n"
        "Rules:\n"
        "- Output EXACTLY one line.\n"
        "- The line must be: DECISION: APPROVE  OR  DECISION: REJECT\n\n"
        "DECISION:"
    )


@dataclass
class GenerationResult:
    decision: str
    raw_text: str
    prompt: str
    logprob: float = 0.0
    approve_probability: float = 0.5


class Agent2LLMPolicy:
    """
    Minimal LLM policy used by both:
      - inference: generate_decision() for the pipeline
      - training: prompt formatting for GRPOTrainer generation
    """

    def __init__(
        self,
        *,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_seq_length: int = 512,
        load_in_4bit: bool = True,
        device: str | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.max_seq_length = int(max_seq_length)
        self.load_in_4bit = bool(load_in_4bit)
        self.device = device

        self._model = None
        self._tokenizer = None

    def build_prompt(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        return format_prompt(features=features, risk_score=risk_score, shap_info=shap_info)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        # Prefer Unsloth (fast path), but allow a pure-Transformers fallback for Windows/CPU setups.
        try:
            from unsloth import FastLanguageModel  # type: ignore
            from transformers import AutoTokenizer  # type: ignore

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name_or_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
            FastLanguageModel.for_inference(model)
        except Exception:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

        self._model = model
        self._tokenizer = tokenizer

    def generate_decision(
        self,
        features: Mapping[str, Any] | Mapping[str, object],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        result = self.generate_with_metadata(features, float(risk_score), shap_info)
        return result.decision

    def generate_with_metadata(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> GenerationResult:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        prompt = self.build_prompt(features=features, risk_score=risk_score, shap_info=shap_info)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generated = self._model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        raw_text = self._tokenizer.decode(generated[0], skip_special_tokens=True)
        completion = raw_text[len(prompt) :] if raw_text.startswith(prompt) else raw_text
        decision = extract_decision(completion)
        if decision == "INVALID":
            # Fail-closed: prefer reject when model violates format.
            decision = "REJECT"
        return GenerationResult(
            decision=decision,
            raw_text=completion.strip(),
            prompt=prompt,
            logprob=0.0,
            approve_probability=0.5,
        )


def default_agent2_checkpoint_dir() -> Path:
    return Path("agent2_llm_checkpoints") / "grpo"

