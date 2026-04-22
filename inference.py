from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.synthetic_generator import generate_synthetic_data
from pipeline.main_pipeline import CreditDecisionPipeline


ROOT = Path(__file__).resolve().parent


def _load_agent2_module():
    module_path = ROOT / "agent2-decision-base" / "train.py"
    module_name = "credless_agent2_inference"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Agent 2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Agent 1 + Agent 2 inference.")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV path.")
    parser.add_argument("--n-rows", type=int, default=256, help="Number of rows to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="inference_results.jsonl", help="Output JSONL path.")
    parser.add_argument("--samples", type=int, default=5, help="How many sample rows to print.")
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


def main() -> None:
    args = parse_args()
    frame, source = _load_frame(args.csv, args.n_rows, args.seed)
    pipeline = CreditDecisionPipeline()
    agent2_module = _load_agent2_module()
    results: list[dict[str, object]] = []

    for record in frame.to_dict(orient="records"):
        output = pipeline.run(record)
        model_output = str(output["policy_output"].get("raw_text", output.get("decision", "")))
        decision = agent2_module.extract_decision(model_output)
        if decision == "INVALID":
            decision = agent2_module.extract_decision(str(output.get("decision", "")))
        if decision == "INVALID":
            decision = "REJECT"

        result = {
            "decision": decision,
            "reward": round(float(output.get("reward", 0.0)), 6),
            "risk_score": round(float(output.get("risk_score", 0.0)), 6),
            "oracle_score": round(float(output.get("info", {}).get("oracle_score", 0.0)), 6),
            "done": True,
            "explanation": str(output.get("info", {}).get("explanation", "")),
            "approve_probability": round(float(output.get("policy_output", {}).get("approve_probability", 0.0)), 6),
        }
        results.append(result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    rewards = np.array([float(item["reward"]) for item in results], dtype=float)
    decisions = [str(item["decision"]) for item in results]
    sample_count = min(args.samples, len(results))
    sample_indices = np.random.default_rng(args.seed).choice(len(results), sample_count, replace=False)

    print("=" * 72)
    print("Frozen Agent 1 + Agent 2 inference")
    print("=" * 72)
    print(f"Rows evaluated              : {len(results)}")
    print(f"Data source                 : {source}")
    print(f"Average reward              : {float(rewards.mean()):.4f}")
    print(f"Reward std                  : {float(rewards.std()):.4f}")
    print(f"Approve rate                : {decisions.count('APPROVE') / len(decisions):.4f}")
    print(f"Output file                 : {output_path}")
    print()
    print("Sample outputs:")
    for idx in sample_indices:
        item = results[int(idx)]
        print(
            f"- row={int(idx)} decision={item['decision']} "
            f"risk_score={float(item['risk_score']):.4f} reward={float(item['reward']):.4f}"
        )


if __name__ == "__main__":
    main()
