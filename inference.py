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
from pipeline.main_pipeline import CreditDecisionEnvironment, CreditDecisionPipeline, _load_agent2_module


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


def _run_one_local(agent1: Any, agent2: Any, record: Mapping[str, Any]) -> dict[str, Any]:
    features = {str(key).strip().lower().replace(" ", "_"): value for key, value in dict(record).items()}
    risk_score = float(agent1.predict(features))
    shap_info = list(agent1.explain(features))

    if hasattr(agent2, "generate_with_metadata"):
        policy_output = agent2.generate_with_metadata(features, risk_score, shap_info)
        decision = str(policy_output.decision)
    else:
        decision = str(agent2.generate_decision(features, risk_score, shap_info))

    env = CreditDecisionEnvironment(features)
    result = env.step(decision)
    return {
        "risk_score": round(risk_score, 6),
        "decision": decision,
        "reward": round(float(result["reward"]), 6),
        "oracle_score": round(float(result["info"].get("oracle_score", 0.0)), 6),
        "explanation": str(result["info"].get("explanation", "")),
        "oracle_decision": str(result["info"].get("oracle_decision", "")),
    }


def _aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.array([float(item["reward"]) for item in results], dtype=float)
    approvals = [str(item["decision"]) for item in results]
    oracle_matches = [
        1.0 if str(item["decision"]) == str(item["oracle_decision"]) else 0.0
        for item in results
    ]
    return {
        "mean_reward": round(float(rewards.mean()), 6) if len(rewards) else 0.0,
        "approve_rate": round(approvals.count("APPROVE") / len(approvals), 6) if approvals else 0.0,
        "oracle_agreement": round(float(np.mean(oracle_matches)), 6) if oracle_matches else 0.0,
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
    for record in frame.to_dict(orient="records"):
        if time.time() - start_time > MAX_RUNTIME_SECONDS:
            raise TimeoutError(f"Inference exceeded the {MAX_RUNTIME_SECONDS}s runtime budget.")
        sample_result = _run_one_local(agent1, agent2, record)
        results.append(sample_result)

    summary = _aggregate_metrics(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            public_item = {
                "risk_score": item["risk_score"],
                "decision": item["decision"],
                "reward": item["reward"],
                "oracle_score": item["oracle_score"],
                "explanation": item["explanation"],
            }
            handle.write(json.dumps(public_item, ensure_ascii=True) + "\n")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
