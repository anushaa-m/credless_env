from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = MODULE_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "agent2_policy.pkl"
METADATA_PATH = CHECKPOINT_DIR / "agent2_metadata.json"
DATASET_EXPORT_PATH = CHECKPOINT_DIR / "supervised_dataset.jsonl"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data.synthetic_generator import generate_synthetic_data
from pipeline.oracle import oracle_decision

DEFAULT_NUMERIC_FIELDS = [
    "age",
    "monthlyincome",
    "debtratio",
    "revolvingutilizationofunsecuredlines",
    "numberoftime30-59dayspastduenotworse",
    "numberoftime60-89dayspastduenotworse",
    "numberoftimes90dayslate",
    "numberofopencreditlinesandloans",
    "numberrealestateloansorlines",
    "years_at_address",
    "months_employed",
    "bank_account_age_months",
    "avg_monthly_balance",
    "negative_balance_days",
    "overdraft_count",
    "monthly_upi_spend",
    "active_txn_days",
    "failed_txn_ratio",
    "late_night_txn_ratio",
    "utility_payment_ratio",
    "rent_payment_regular",
    "emi_payment_ratio",
    "vehicle_owned",
    "gold_value_estimate",
    "fd_amount",
    "emergency_savings_months",
    "property_owned",
    "income_variability_score",
    "location_risk_index",
    "current_medical_condition",
    "medical_debt",
    "gov_scheme_enrollment",
    "business_age",
    "business_type_risk",
    "monthly_revenue",
    "profit_margin",
    "numberofdependents",
    "dependents_count",
]


def extract_decision(text: str) -> str:
    text = text.upper()
    if "APPROVE" in text:
        return "APPROVE"
    if "REJECT" in text:
        return "REJECT"
    return "INVALID"


def _normalize_feature_dict(features: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(features).items():
        if isinstance(value, np.generic):
            value = value.item()
        normalized[str(key).strip().lower().replace(" ", "_")] = value
    return normalized


def _load_pipeline_module():
    module_path = ROOT / "pipeline" / "main_pipeline.py"
    module_name = "credless_pipeline_main"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load pipeline module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def format_prompt(
    features: Mapping[str, Any],
    risk_score: float,
    shap_info: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    features = _normalize_feature_dict(features)
    shap_items = list(shap_info or [])

    def _value(name: str, default: Any = "N/A") -> Any:
        return features.get(name, default)

    top_lines = []
    for item in shap_items[:3]:
        feature_name = str(item.get("feature", item.get("name", "unknown")))
        impact = item.get("impact", item.get("direction", item.get("contribution", "unknown")))
        contribution = item.get("contribution")
        if contribution is None:
            top_lines.append(f"- {feature_name}: {impact}")
        else:
            top_lines.append(f"- {feature_name}: {impact} ({float(contribution):+.4f})")

    if not top_lines:
        top_lines.append("- risk_score: model confidence is the strongest available signal")

    return (
        "User Profile:\n"
        f"- Age: {_value('age')}\n"
        f"- Income: {_value('monthlyincome')}\n"
        f"- Credit Score: {_value('credit_score', _value('numberofopencreditlinesandloans'))}\n"
        f"- Risk Score: {float(risk_score):.6f}\n"
        "\n"
        "Top Risk Factors:\n"
        f"{chr(10).join(top_lines)}\n"
        "\n"
        "Respond with ONLY one word:\n"
        "APPROVE or REJECT"
    )


@dataclass
class Agent2Config:
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    backend: str = "lightweight"
    seed: int = 42
    learning_rate: float = 0.03
    max_iter: int = 2000
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    numeric_fields: list[str] | None = None

    def resolved_numeric_fields(self) -> list[str]:
        return list(self.numeric_fields or DEFAULT_NUMERIC_FIELDS)


@dataclass
class GenerationResult:
    decision: str
    raw_text: str
    prompt: str
    logprob: float
    approve_probability: float


class Agent2DecisionMaker:
    def __init__(
        self,
        *,
        config: Agent2Config | None = None,
        classifier: SGDClassifier | None = None,
        numeric_fields: Sequence[str] | None = None,
        feature_stats: Mapping[str, Mapping[str, float]] | None = None,
        trained: bool = False,
        forced_label: int | None = None,
    ) -> None:
        self.config = config or Agent2Config()
        self.numeric_fields = list(numeric_fields or self.config.resolved_numeric_fields())
        self.classifier = classifier
        self.feature_stats = {
            name: {"mean": float(stats.get("mean", 0.0)), "std": max(float(stats.get("std", 1.0)), 1e-6)}
            for name, stats in (feature_stats or {}).items()
        }
        self.trained = trained
        self.forced_label = forced_label
        self._classes = np.array([0, 1], dtype=int)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path | None = None) -> "Agent2DecisionMaker":
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
        model_path = checkpoint_root / "agent2_policy.pkl"
        if not model_path.exists():
            return cls(config=Agent2Config(checkpoint_dir=str(checkpoint_root)))

        with open(model_path, "rb") as handle:
            payload = pickle.load(handle)

        config = Agent2Config(**payload.get("config", {}))
        return cls(
            config=config,
            classifier=payload.get("classifier"),
            numeric_fields=payload.get("numeric_fields"),
            feature_stats=payload.get("feature_stats"),
            trained=bool(payload.get("trained", False)),
            forced_label=payload.get("forced_label"),
        )

    def save(self, checkpoint_dir: str | Path | None = None) -> None:
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else CHECKPOINT_DIR
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "numeric_fields": self.numeric_fields,
            "feature_stats": self.feature_stats,
            "classifier": self.classifier,
            "trained": self.trained,
            "forced_label": self.forced_label,
        }
        with open(checkpoint_root / "agent2_policy.pkl", "wb") as handle:
            pickle.dump(payload, handle)
        metadata = {
            "backend": self.config.backend,
            "base_model_name": self.config.base_model_name,
            "trained": self.trained,
            "numeric_fields": self.numeric_fields,
            "checkpoint_dir": str(checkpoint_root).replace("\\", "/"),
        }
        (checkpoint_root / "agent2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def fit_supervised(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        validation_size: float = 0.20,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("No training records were provided for Agent 2.")

        frame = pd.DataFrame(records)
        if "label" not in frame.columns:
            raise ValueError("Training records must contain a 'label' column.")

        labels = frame["label"].astype(int).to_numpy()
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            self.classifier = None
            self.forced_label = int(unique_labels[0])
            self.trained = True
            metrics = {
                "validation_accuracy": 1.0,
                "n_train": int(len(labels)),
                "n_val": 0,
                "classification_report": f"Single-class dataset detected. Persisted constant policy: {self.forced_label}.",
            }
            return metrics

        vectors = np.vstack(
            [
                self.vectorize(
                    features=row["features"],
                    risk_score=float(row["risk_score"]),
                    shap_info=row.get("shap_info"),
                )
                for _, row in frame.iterrows()
            ]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            vectors,
            labels,
            test_size=validation_size,
            random_state=random_state or self.config.seed,
            stratify=labels,
        )

        self.classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.config.learning_rate,
            max_iter=self.config.max_iter,
            class_weight="balanced",
            random_state=random_state or self.config.seed,
        )
        self.classifier.fit(X_train, y_train)
        self.trained = True
        self.forced_label = None

        val_prob = self.classifier.predict_proba(X_val)[:, 1]
        val_pred = (val_prob >= 0.50).astype(int)
        metrics = {
            "validation_accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "classification_report": classification_report(
                y_val,
                val_pred,
                target_names=["REJECT", "APPROVE"],
                zero_division=0,
            ),
        }
        return metrics

    def partial_fit_from_feedback(self, samples: Sequence[Mapping[str, Any]]) -> None:
        if not samples:
            return

        vectors = np.vstack(
            [
                self.vectorize(
                    features=sample["features"],
                    risk_score=float(sample["risk_score"]),
                    shap_info=sample.get("shap_info"),
                )
                for sample in samples
            ]
        )
        labels = np.array([1 if str(sample["label"]).upper() == "APPROVE" else 0 for sample in samples], dtype=int)
        weights = np.array([max(float(sample.get("weight", 1.0)), 1e-3) for sample in samples], dtype=float)
        unique_labels = np.unique(labels)

        if len(unique_labels) < 2 and self.classifier is None:
            self.forced_label = int(unique_labels[0])
            self.trained = True
            return

        if len(unique_labels) < 2:
            opposite_label = 1 - int(unique_labels[0])
            vectors = np.vstack([vectors, vectors[[0]]])
            labels = np.append(labels, opposite_label)
            weights = np.append(weights, 1e-6)

        if self.classifier is None:
            self.classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.config.learning_rate,
                max_iter=1,
                warm_start=True,
                class_weight="balanced",
                random_state=self.config.seed,
            )
            self.classifier.partial_fit(vectors, labels, classes=self._classes, sample_weight=weights)
        else:
            self.classifier.partial_fit(vectors, labels, classes=self._classes, sample_weight=weights)
        self.trained = True
        self.forced_label = None

    def _heuristic_approve_probability(
        self,
        *,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        features = _normalize_feature_dict(features)
        approve_prob = 1.0 - float(risk_score)
        approve_prob += 0.06 if float(features.get("utility_payment_ratio", 0) or 0) >= 0.90 else 0.0
        approve_prob -= 0.10 if float(features.get("failed_txn_ratio", 0) or 0) >= 0.35 else 0.0
        approve_prob -= 0.12 if float(features.get("numberoftimes90dayslate", 0) or 0) >= 1 else 0.0
        approve_prob += 0.05 if float(features.get("emergency_savings_months", 0) or 0) >= 3 else 0.0

        for item in shap_info or []:
            contribution = float(item.get("contribution", 0.0) or 0.0)
            approve_prob -= 0.03 * np.sign(contribution)

        return float(np.clip(approve_prob, 0.01, 0.99))

    def vectorize(
        self,
        *,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> np.ndarray:
        values = _normalize_feature_dict(features)
        vector: list[float] = [float(risk_score)]

        for field in self.numeric_fields:
            raw_value = values.get(field, 0.0)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = 0.0

            stats = self.feature_stats.get(field)
            if stats is None:
                vector.append(numeric_value)
            else:
                vector.append((numeric_value - stats["mean"]) / stats["std"])

        contributions = [float(item.get("contribution", 0.0) or 0.0) for item in shap_info or []]
        vector.append(float(np.mean(contributions)) if contributions else 0.0)
        vector.append(float(np.max(contributions)) if contributions else 0.0)
        vector.append(float(np.min(contributions)) if contributions else 0.0)
        vector.append(float(len(contributions)))
        return np.asarray(vector, dtype=float)

    def predict_approve_probability(
        self,
        *,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        if self.forced_label is not None:
            return 0.99 if self.forced_label == 1 else 0.01
        if self.classifier is None or not self.trained:
            return self._heuristic_approve_probability(features=features, risk_score=risk_score, shap_info=shap_info)

        vector = self.vectorize(features=features, risk_score=risk_score, shap_info=shap_info).reshape(1, -1)
        probability = float(self.classifier.predict_proba(vector)[0][1])
        return float(np.clip(probability, 0.01, 0.99))

    def generate_with_metadata(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> GenerationResult:
        prompt = format_prompt(features, risk_score, shap_info)
        approve_probability = self.predict_approve_probability(
            features=features,
            risk_score=risk_score,
            shap_info=shap_info,
        )
        raw_text = "APPROVE" if approve_probability >= 0.50 else "REJECT"
        decision = extract_decision(raw_text)
        if decision == "INVALID":
            raw_text = "REJECT"
            decision = "REJECT"
            approve_probability = min(approve_probability, 0.49)

        chosen_probability = approve_probability if decision == "APPROVE" else 1.0 - approve_probability
        logprob = math.log(max(chosen_probability, 1e-8))
        return GenerationResult(
            decision=decision,
            raw_text=raw_text,
            prompt=prompt,
            logprob=float(logprob),
            approve_probability=float(approve_probability),
        )

    def generate_decision(
        self,
        features: Mapping[str, Any],
        risk_score: float,
        shap_info: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        return self.generate_with_metadata(features, risk_score, shap_info).decision


def _load_agent1_predictor():
    pipeline_module = _load_pipeline_module()
    return pipeline_module.FrozenRiskPredictor()


def _load_dataset(csv_path: str | None, n_rows: int | None, seed: int) -> pd.DataFrame:
    if csv_path:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Training CSV not found at {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
        frame = frame.loc[:, ~frame.columns.duplicated()]
    else:
        frame = generate_synthetic_data(
            n_samples=n_rows or 4096,
            seed=seed,
            include_target=True,
        )

    if n_rows is not None and n_rows > 0 and len(frame) > n_rows:
        frame = frame.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _infer_feature_stats(frame: pd.DataFrame, numeric_fields: Sequence[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field in numeric_fields:
        if field not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[field], errors="coerce").fillna(0.0)
        std = float(numeric.std(ddof=0))
        stats[field] = {
            "mean": float(numeric.mean()),
            "std": std if std > 1e-6 else 1.0,
        }
    return stats


def build_supervised_records(
    frame: pd.DataFrame,
    *,
    predictor: Any | None = None,
) -> list[dict[str, Any]]:
    predictor = predictor or _load_agent1_predictor()
    records: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        features = _normalize_feature_dict(row.to_dict())
        risk_score = float(predictor.predict(features))
        shap_info = predictor.explain(features)
        oracle = oracle_decision(features)
        label = 1 if str(oracle["decision"]).lower() == "approve" else 0

        records.append(
            {
                "features": features,
                "risk_score": risk_score,
                "shap_info": shap_info,
                "prompt": format_prompt(features, risk_score, shap_info),
                "target_text": "APPROVE" if label == 1 else "REJECT",
                "label": label,
            }
        )
    return records


def export_supervised_records(records: Sequence[Mapping[str, Any]], output_path: str | Path = DATASET_EXPORT_PATH) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def generate_decision(
    features: Mapping[str, Any],
    risk_score: float,
    shap_info: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    return Agent2DecisionMaker.from_checkpoint().generate_decision(features, risk_score, shap_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Agent 2 decision policy.")
    parser.add_argument("--csv", type=str, default=None, help="Optional training CSV path.")
    parser.add_argument("--n-rows", type=int, default=4096, help="Max rows to train on.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument(
        "--backend",
        type=str,
        default="lightweight",
        choices=["lightweight"],
        help="Current training backend. Prompt records are still exported for future HF/TRL training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _load_dataset(args.csv, args.n_rows, args.seed)
    config = Agent2Config(backend=args.backend, seed=args.seed)
    config.numeric_fields = DEFAULT_NUMERIC_FIELDS

    predictor = _load_agent1_predictor()
    records = build_supervised_records(frame, predictor=predictor)
    export_supervised_records(records)

    agent = Agent2DecisionMaker(
        config=config,
        numeric_fields=config.resolved_numeric_fields(),
        feature_stats=_infer_feature_stats(frame, config.resolved_numeric_fields()),
    )
    metrics = agent.fit_supervised(
        records,
        validation_size=args.validation_size,
        random_state=args.seed,
    )
    agent.save()

    print("=" * 72)
    print("Agent 2 supervised training complete")
    print("=" * 72)
    print(f"Rows used                  : {len(records)}")
    print(f"Validation accuracy        : {metrics['validation_accuracy']:.4f}")
    print(f"Checkpoint                 : {MODEL_PATH}")
    print(f"Prompt dataset export      : {DATASET_EXPORT_PATH}")
    print()
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
