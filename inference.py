from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_data
from pipeline.main_pipeline import CreditDecisionPipeline, _load_agent2_module
from server.environment import CreditAnalystEnvironment


DEFAULT_SEED = int(os.getenv("CREDLESS_SEED", "42"))
DEFAULT_ROWS = int(os.getenv("CREDLESS_N_ROWS", "256"))
DEFAULT_CSV = os.getenv("CREDLESS_CSV")
DEFAULT_OUTPUT = os.getenv("CREDLESS_OUTPUT", "inference_results.jsonl")
DEFAULT_SUMMARY_OUTPUT = os.getenv("CREDLESS_SUMMARY_OUTPUT", "inference_summary.json")
DEFAULT_AGENT2_BACKEND = os.getenv("CREDLESS_AGENT2_BACKEND", "local").strip().lower()
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
MAX_RUNTIME_SECONDS = 20 * 60
MAX_EPISODE_STEPS = 8
APPROVE_THRESHOLD = 0.30
LOW_RISK_AUTO_APPROVE = 0.40
LOW_RISK_HEURISTIC_CAP = 0.50
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


def _oracle_factor_names(top_factors: list[list[Any]]) -> list[str]:
    names: list[str] = []
    for item in top_factors:
        if isinstance(item, list) and item:
            names.append(str(item[0]))
    return names


def _normalize_decision_label(value: Any) -> str:
    decision = str(value or "").strip().upper()
    if decision == "REJECT":
        return "DENY"
    return decision


def _oracle_agreed(decision: Any, oracle_decision: Any) -> bool:
    return _normalize_decision_label(decision) == _normalize_decision_label(oracle_decision)


def _local_oracle_payload(env: CreditAnalystEnvironment) -> dict[str, Any]:
    revealed = env.oracle_features()
    merged = {field: 0.5 for field in getattr(env.oracle, "feature_order", [])}
    merged.update(revealed)
    market_state = dict(env._market_state)  # type: ignore[attr-defined]
    oracle_result = env.oracle.predict(merged, market_condition=market_state["name"])
    oracle_risk = float(oracle_result.get("default_prob", 0.0))
    oracle_confidence = float(max(oracle_risk, 1.0 - oracle_risk))
    thresholds = dict(oracle_result.get("thresholds", {}))
    top_factors = [
        [str(item["feature"]), float(item["contribution"])]
        for item in env.risk_predictor.explain(merged, top_k=5)
    ]
    return {
        "oracle_risk": oracle_risk,
        "oracle_confidence": oracle_confidence,
        "market_state": market_state,
        "base_threshold": thresholds.get("base_medium_risk"),
        "dynamic_threshold": thresholds.get("dynamic_threshold", thresholds.get("medium_risk")),
        "market_risk_index": thresholds.get("market_risk_index"),
        "top_factors": top_factors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic CredLess inference runner.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Optional CSV path.")
    parser.add_argument("--n-rows", type=int, default=DEFAULT_ROWS, help="Number of rows to evaluate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Per-sample JSONL output path.")
    parser.add_argument(
        "--summary-output",
        type=str,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Aggregate metrics JSON output path.",
    )
    parser.add_argument(
        "--agent2-backend",
        type=str,
        choices=["local", "openai"],
        default=DEFAULT_AGENT2_BACKEND,
        help="Use the local Agent 2 policy or an OpenAI-compatible chat backend.",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Remote model name for openai backend.")
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=DEFAULT_API_BASE_URL,
        help="Optional OpenAI-compatible base URL for remote Agent 2 inference.",
    )
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key for the remote backend.")
    return parser.parse_args()


def _load_frame(csv_path: str | None, n_rows: int, seed: int) -> tuple[pd.DataFrame, str]:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found at {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
        if n_rows > 0 and len(frame) > n_rows:
            frame = frame.sample(n=n_rows, random_state=seed).reset_index(drop=True)
        source = str(path)
    else:
        frame = generate_synthetic_data(n_samples=n_rows, seed=seed, include_target=False)
        source = "synthetic"
    return frame.reset_index(drop=True), source


class OpenAICompatibleAgent2:
    def __init__(self, *, model_name: str, api_base_url: str | None, api_key: str | None) -> None:
        if not api_key:
            raise ValueError("The openai backend requires OPENAI_API_KEY or HF_TOKEN.")
        from openai import OpenAI

        self._client = OpenAI(base_url=api_base_url, api_key=api_key)
        self._model_name = model_name
        self._module = _load_agent2_module()

    def generate_decision(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: list[dict[str, Any]],
    ) -> str:
        prompt = self._module.format_prompt(features, risk_score, shap_info)
        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4,
        )
        content = (completion.choices[0].message.content or "").strip()
        decision = self._module.extract_decision(content)
        if decision not in {"APPROVE", "REJECT"}:
            raise ValueError(f"Remote Agent 2 returned invalid action: {content!r}")
        return decision


def _build_agent2(args: argparse.Namespace, pipeline: CreditDecisionPipeline) -> Any:
    if args.agent2_backend == "openai":
        return OpenAICompatibleAgent2(
            model_name=args.model_name,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
        )
    return pipeline.agent2


def _profile_features(observation: Mapping[str, Any], *, defaults: Mapping[str, float] | None = None) -> dict[str, float]:
    features = dict(defaults or {})
    profile = observation.get("applicant", {}).get("profile", {})
    for field, payload in profile.items():
        features[str(field)] = float(payload.get("value", 0.0))
    return features


def _choose_action(observation: Mapping[str, Any], oracle_payload: Mapping[str, Any], agent1: Any, agent2: Any) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    missing_fields = list(observation.get("applicant", {}).get("missing_fields", []))
    required_fields = list((observation.get("current_policy") or {}).get("required_fields", []))
    profile = observation.get("applicant", {}).get("profile", {})
    oracle_risk = float(oracle_payload.get("oracle_risk", 0.0) or 0.0)
    oracle_confidence = float(oracle_payload.get("oracle_confidence", 0.0) or 0.0)
    factor_names = _oracle_factor_names(list(oracle_payload.get("top_factors", []) or []))

    for field in required_fields:
        if field in missing_fields:
            return {"action_type": "request_info", "params": {"field": field}}, 0.0, []

    for field in factor_names:
        if field in missing_fields and field not in profile:
            return {"action_type": "request_info", "params": {"field": field}}, 0.0, []

    for field in REQUEST_PRIORITY:
        if field in missing_fields and field not in profile:
            return {"action_type": "request_info", "params": {"field": field}}, 0.0, []

    if not observation.get("market_visible", False) and oracle_confidence < 0.85:
        return {"action_type": "query_market"}, 0.0, []

    confidence = {
        key: float(value.get("confidence", 0.0))
        for key, value in profile.items()
        if isinstance(value, Mapping)
    }
    if (
        confidence and any((1.0 - value) > 0.25 for value in confidence.values())
        and oracle_risk > 0.88
    ) and not observation.get("fraud_flags_raised"):
        return {
            "action_type": "flag_fraud",
            "params": {"reason": "low-confidence profile values or elevated oracle risk require manual verification"},
            "reasoning": "Potential fraud indicators identified from observed applicant data and oracle risk.",
        }, 0.0, []

    current_step = int(observation.get("step", 0) or 0)
    max_steps = int(observation.get("max_steps", MAX_EPISODE_STEPS) or MAX_EPISODE_STEPS)
    if missing_fields and current_step <= max_steps - 3 and (oracle_risk > 0.55 or oracle_confidence < 0.75):
        for field in REQUEST_PRIORITY:
            if field in missing_fields and field not in profile:
                return {"action_type": "request_info", "params": {"field": field}}, 0.0, []
        return {"action_type": "request_info", "params": {"field": str(missing_fields[0])}}, 0.0, []

    features = _profile_features(observation, defaults={field: 0.5 for field in getattr(agent1, "feature_order", [])})
    heuristic_risk = float(agent1.predict(features))
    risk_score = 0.60 * oracle_risk + 0.40 * heuristic_risk if oracle_risk > 0.0 else heuristic_risk
    shap_info = list(agent1.explain(features))

    policy_approve_probability: float | None = None
    if hasattr(agent2, "generate_with_metadata"):
        policy_output = agent2.generate_with_metadata(features, risk_score, shap_info)
        decision = str(policy_output.decision).strip().upper()
        reasoning = getattr(policy_output, "raw_text", "") or "Decision from agent 2 policy."
        raw_probability = getattr(policy_output, "approve_probability", None)
        if raw_probability is not None:
            policy_approve_probability = float(np.clip(float(raw_probability), 0.01, 0.99))
    else:
        decision = str(agent2.generate_decision(features, risk_score, shap_info)).strip().upper()
        reasoning = "Decision from agent 2 policy."

    if observation.get("task_name") == "adaptive_inquiry" and oracle_confidence < 0.60:
        return {
            "action_type": "escalate",
            "reasoning": f"Escalating because oracle confidence is only {oracle_confidence:.3f}.",
        }, risk_score, shap_info

    heuristic_approve_probability = float(np.clip(1.0 - risk_score, 0.01, 0.99))
    oracle_approve_probability = float(np.clip(1.0 - oracle_risk, 0.01, 0.99))
    if policy_approve_probability is None:
        policy_approve_probability = 0.55 if decision == "APPROVE" else 0.45

    blended_approve_probability = float(
        np.clip(
            0.35 * policy_approve_probability + 0.45 * oracle_approve_probability + 0.20 * heuristic_approve_probability,
            0.01,
            0.99,
        )
    )

    auto_approve = (
        oracle_risk <= LOW_RISK_AUTO_APPROVE
        and risk_score <= LOW_RISK_HEURISTIC_CAP
        and not observation.get("fraud_flags_raised")
    )
    terminal_action = "approve" if auto_approve or blended_approve_probability >= APPROVE_THRESHOLD else "deny"
    return {
        "action_type": terminal_action,
        "reasoning": (
            f"{reasoning} Oracle risk={oracle_risk:.3f}, confidence={oracle_confidence:.3f}, "
            f"approve_prob={blended_approve_probability:.3f}."
        ),
    }, risk_score, shap_info


def _run_one_local(agent1: Any, agent2: Any, episode_seed: int) -> dict[str, Any]:
    env = CreditAnalystEnvironment()
    observation = env.reset(seed=episode_seed)
    oracle_payload = _local_oracle_payload(env)
    done = bool(observation.get("done", False))
    risk_score = 0.0
    shap_info: list[dict[str, Any]] = []
    final_result: dict[str, Any] = {"reward": 0.0, "done": done, "info": {}}
    final_action = "ESCALATE"

    steps = 0
    while not done and steps < MAX_EPISODE_STEPS:
        steps += 1
        action, maybe_risk, maybe_shap = _choose_action(observation, oracle_payload, agent1, agent2)
        if maybe_shap:
            risk_score = maybe_risk
            shap_info = maybe_shap
            final_action = str(action["action_type"]).upper()
        result = env.step(action)
        observation = result.get("observation", observation)
        oracle_payload = _local_oracle_payload(env)
        done = bool(result.get("done", False))
        final_result = result

    return {
        "risk_score": round(risk_score, 6),
        "oracle_risk": round(float(oracle_payload.get("oracle_risk", 0.0)), 6),
        "oracle_confidence": round(float(oracle_payload.get("oracle_confidence", 0.0)), 6),
        "market_state": dict(oracle_payload.get("market_state", {})),
        "base_threshold": final_result["info"].get("base_threshold", oracle_payload.get("base_threshold")),
        "dynamic_threshold": final_result["info"].get("dynamic_threshold", oracle_payload.get("dynamic_threshold")),
        "market_risk_index": final_result["info"].get("market_risk_index", oracle_payload.get("market_risk_index")),
        "top_factors": list(oracle_payload.get("top_factors", [])),
        "decision": final_action,
        "reward": round(float(final_result["reward"]), 6),
        "oracle_score": round(float(final_result["info"].get("oracle_score", 0.0)), 6),
        "explanation": str(final_result["info"].get("explanation", "")),
        "oracle_decision": str(final_result["info"].get("oracle_decision", "")),
        "oracle_agreed": _oracle_agreed(final_action, final_result["info"].get("oracle_decision", "")),
        "steps": steps,
        "action_history": list(observation.get("action_history", [])),
    }


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.array([float(item["reward"]) for item in results], dtype=float)
    approvals = [str(item["decision"]) for item in results]
    oracle_matches = [1.0 if _oracle_agreed(item["decision"], item["oracle_decision"]) else 0.0 for item in results]
    dynamic_thresholds = [
        float(item["dynamic_threshold"])
        for item in results
        if item.get("dynamic_threshold") is not None
    ]
    market_counts: dict[str, int] = {}
    for item in results:
        market_name = str(dict(item.get("market_state", {})).get("name", "unknown"))
        market_counts[market_name] = market_counts.get(market_name, 0) + 1
    return {
        "mean_reward": round(float(rewards.mean()), 6) if len(rewards) else 0.0,
        "approve_rate": round(approvals.count("APPROVE") / len(approvals), 6) if approvals else 0.0,
        "oracle_agreement": round(float(np.mean(oracle_matches)), 6) if oracle_matches else 0.0,
        "oracle_agreement_count": int(sum(oracle_matches)),
        "mean_dynamic_threshold": round(float(np.mean(dynamic_thresholds)), 6) if dynamic_thresholds else 0.0,
        "market_counts": market_counts,
        "episodes": len(results),
    }


def main() -> None:
    args = parse_args()
    start_time = time.time()
    np.random.seed(args.seed)

    frame, _ = _load_frame(args.csv, args.n_rows, args.seed)
    pipeline = CreditDecisionPipeline()
    agent1 = pipeline.agent1
    agent2 = _build_agent2(args, pipeline)

    results: list[dict[str, Any]] = []
    for episode_index, _record in enumerate(frame.to_dict(orient="records")):
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            raise TimeoutError(f"Inference exceeded the {MAX_RUNTIME_SECONDS}s runtime budget.")
        sample_result = _run_one_local(agent1, agent2, args.seed + episode_index)
        results.append(sample_result)

    summary = _aggregate_metrics(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            public_item = {
                "risk_score": item["risk_score"],
                "oracle_risk": item["oracle_risk"],
                "oracle_confidence": item["oracle_confidence"],
                "market_state": item["market_state"],
                "base_threshold": item["base_threshold"],
                "dynamic_threshold": item["dynamic_threshold"],
                "market_risk_index": item["market_risk_index"],
                "top_factors": item["top_factors"],
                "decision": item["decision"],
                "reward": item["reward"],
                "oracle_score": item["oracle_score"],
                "oracle_decision": item["oracle_decision"],
                "oracle_agreed": item["oracle_agreed"],
                "explanation": item["explanation"],
            }
            handle.write(json.dumps(public_item, ensure_ascii=True) + "\n")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
