from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


MODEL_PATH = Path("models/saved/finverse_model.pkl")
META_PATH = Path("models/saved/model_meta.json")
SCALER_PATH = Path("models/saved/scaler.pkl")


@dataclass
class TrainingArtifacts:
    model: object
    scaler: StandardScaler
    feature_cols: list[str]
    val_accuracy: float
    val_auc: float
    y_val: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    classification_report_text: str


def prob_to_tier(prob: float) -> str:
    if prob >= 0.70:
        return "A"
    if prob >= 0.45:
        return "B"
    return "C"


def prob_to_decision(prob: float) -> str:
    return "approve" if prob >= 0.50 else "reject"


def build_model(seed: int):
    if HAS_LGBM:
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )

    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def save_artifacts(
    *,
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    y_train: np.ndarray,
    y_val: np.ndarray,
    val_acc: float,
    val_auc: float,
    seed: int,
    jsonl_path: str,
) -> None:
    os.makedirs(MODEL_PATH.parent, exist_ok=True)

    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(scaler, handle)

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "model_type": "lightgbm" if HAS_LGBM else "logistic_regression",
        "val_accuracy": round(float(val_acc), 4),
        "val_auc": round(float(val_auc), 4),
        "seed": int(seed),
    }
    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(artifact, handle)

    metadata = {
        "feature_cols": feature_cols,
        "model_type": artifact["model_type"],
        "val_accuracy": round(float(val_acc), 4),
        "val_auc": round(float(val_auc), 4),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "approve_rate_train": round(float(np.mean(y_train)), 4),
        "target_positive_class": "approve",
        "jsonl_path": jsonl_path,
        "model_path": str(MODEL_PATH).replace("\\", "/"),
        "scaler_path": str(SCALER_PATH).replace("\\", "/"),
    }
    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def train_from_frame(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    val_size: float = 0.20,
    jsonl_path: str = "data/dataset.jsonl",
    save: bool = True,
) -> TrainingArtifacts:
    X = df_model[feature_cols].values
    y = df_model["target"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_model(seed)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_val_scaled)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    val_acc = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, target_names=["reject", "approve"])

    if save:
        save_artifacts(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            y_train=y_train,
            y_val=y_val,
            val_acc=val_acc,
            val_auc=val_auc,
            seed=seed,
            jsonl_path=jsonl_path,
        )

    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        val_accuracy=float(val_acc),
        val_auc=float(val_auc),
        y_val=y_val,
        y_prob=y_prob,
        y_pred=y_pred,
        classification_report_text=report,
    )


def load_saved_model() -> tuple[object, StandardScaler, list[str]]:
    with open(MODEL_PATH, "rb") as handle:
        artifact = pickle.load(handle)
    with open(SCALER_PATH, "rb") as handle:
        scaler = pickle.load(handle)

    if not isinstance(artifact, dict):
        raise ValueError("Unexpected saved model format")
    return artifact["model"], scaler, artifact["feature_cols"]
