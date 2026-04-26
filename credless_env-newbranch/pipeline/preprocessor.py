"""
Data preprocessor for the FinVerse-compatible pipeline in this repo.

Behavior:
  - existing CSV path -> load real data
  - missing CSV path  -> synthetic fallback
  - no CSV path       -> synthetic fallback

Real bundled data is still available explicitly at `data/cd_updated.csv`.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from credless_model.dataset_pipeline import DEFAULT_DATASET_PATH
from data.synthetic_generator import generate_synthetic_data
from pipeline.oracle import oracle_decision
from pipeline.reasoning import generate_reasoning

warnings.filterwarnings("ignore")

FEATURE_GROUPS = {
    "financials": [
        "monthlyincome",
        "debtratio",
        "revolvingutilizationofunsecuredlines",
        "avg_monthly_balance",
        "monthly_upi_spend",
        "monthly_revenue",
        "profit_margin",
        "income_variability_score",
    ],
    "credit_behavior": [
        "numberoftime30-59dayspastduenotworse",
        "numberoftime60-89dayspastduenotworse",
        "numberoftimes90dayslate",
        "numberofopencreditlinesandloans",
        "numberrealestateloansorlines",
    ],
    "transaction_behavior": [
        "active_txn_days",
        "failed_txn_ratio",
        "late_night_txn_ratio",
        "negative_balance_days",
        "overdraft_count",
        "bank_account_age_months",
    ],
    "assets": [
        "property_owned",
        "vehicle_owned",
        "gold_value_estimate",
        "fd_amount",
        "emergency_savings_months",
    ],
    "risk_flags": [
        "location_risk_index",
        "income_variability_score",
        "current_medical_condition",
        "medical_debt",
        "business_type_risk",
        "gov_scheme_enrollment",
    ],
    "payment_discipline": [
        "utility_payment_ratio",
        "rent_payment_regular",
        "emi_payment_ratio",
    ],
    "stability": [
        "age",
        "months_employed",
        "years_at_address",
        "marital_status",
        "employment_type",
        "numberofdependents",
        "business_age",
    ],
}

CATEGORICAL_COLS = ["marital_status", "employment_type"]

NUMERIC_COLS = [
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

_MEDIAN_FILL_COLS = [
    "monthlyincome",
    "debtratio",
    "numberofdependents",
    "avg_monthly_balance",
    "gold_value_estimate",
    "fd_amount",
    "monthly_revenue",
    "profit_margin",
    "medical_debt",
    "business_age",
]
_ZERO_FILL_COLS = [
    "numberoftime30-59dayspastduenotworse",
    "numberoftime60-89dayspastduenotworse",
    "numberoftimes90dayslate",
    "overdraft_count",
    "negative_balance_days",
    "vehicle_owned",
    "property_owned",
    "current_medical_condition",
    "gov_scheme_enrollment",
    "rent_payment_regular",
    "business_type_risk",
]


def _approve_target_from_raw(value: object) -> int:
    """
    Convert the dataset target into FinVerse approval semantics.

    The bundled dataset uses `1` for default / reject and `0` for repay.
    FinVerse pipeline targets use `1` for approve and `0` for reject.
    """
    return 0 if int(value) == 1 else 1


def _approve_target_from_value(value: object) -> int:
    return int(value)


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in _MEDIAN_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in _ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_enc"] = pd.Categorical(df[col]).codes
    return df


def _clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    return df


def _row_to_structured_json(row: pd.Series) -> dict:
    structured = {}
    for group, cols in FEATURE_GROUPS.items():
        group_dict = {}
        for col in cols:
            if col in row.index:
                val = row[col]
                if isinstance(val, np.integer):
                    val = int(val)
                elif isinstance(val, np.floating):
                    val = float(round(val, 4))
                group_dict[col] = val
        structured[group] = group_dict
    return structured


def load_and_preprocess(
    csv_path: Optional[str] = None,
    n_synthetic: int = 12000,
    max_rows: Optional[int] = None,
    output_jsonl: Optional[str] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, None]:
    target_converter = _approve_target_from_raw

    if csv_path and Path(csv_path).exists():
        dataset_path = Path(csv_path)
        print(f"[preprocessor] Loading real data from: {csv_path}")
        df = pd.read_csv(dataset_path, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        # The dataset contains a duplicate `property_owned` column; keep the first.
        df = df.loc[:, ~df.columns.duplicated()]
        if max_rows is not None and max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
            print(f"[preprocessor] Sampled {len(df)} rows from real CSV")
    else:
        dataset_path = None
        target_converter = _approve_target_from_value
        if csv_path:
            print(f"[preprocessor] CSV not found at '{csv_path}' - using synthetic fallback")
        else:
            print("[preprocessor] No CSV provided - generating synthetic fallback data")
        df = generate_synthetic_data(
            n_samples=n_synthetic,
            seed=seed,
            include_target=True,
        )

    if dataset_path is None:
        pass

    df = _fill_missing(df)
    df = _clip_outliers(df)
    df = _encode_categoricals(df)

    oracle_results = []
    reasonings = []
    for _, row in df.iterrows():
        result = oracle_decision(row.to_dict())
        oracle_results.append(result)
        reasonings.append(generate_reasoning(row.to_dict(), result))

    df["oracle_decision"] = [r["decision"] for r in oracle_results]
    df["oracle_tier"] = [r["risk_tier"] for r in oracle_results]
    df["oracle_conf"] = [r["confidence"] for r in oracle_results]
    df["reasoning"] = reasonings

    if output_jsonl:
        os.makedirs(Path(output_jsonl).parent, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as handle:
            for i, (_, row) in enumerate(df.iterrows()):
                oracle_res = oracle_results[i]
                record = {
                    "input": _row_to_structured_json(row),
                    "output": {
                        "decision": "reject" if oracle_res["decision"] == "deny" else oracle_res["decision"],
                        "risk_tier": oracle_res["risk_tier"],
                        "confidence": oracle_res["confidence"],
                        "reasoning": reasonings[i],
                    },
                    "target": (
                        target_converter(row["target"])
                        if "target" in row.index
                        else int(oracle_res["decision"] == "approve")
                    ),
                }
                handle.write(json.dumps(record) + "\n")
        print(f"[preprocessor] dataset.jsonl written -> {output_jsonl}")

    ml_features = [col for col in NUMERIC_COLS if col in df.columns]
    for col in CATEGORICAL_COLS:
        enc_col = f"{col}_enc"
        if enc_col in df.columns:
            ml_features.append(enc_col)
    ml_features = list(dict.fromkeys(ml_features))

    df_model = df[ml_features].copy()
    if "target" in df.columns:
        df_model["target"] = df["target"].apply(target_converter).astype(int).values
    else:
        df_model["target"] = (df["oracle_decision"] == "approve").astype(int).values

    print(
        f"[preprocessor] ML DataFrame: {df_model.shape} | "
        f"Approve rate: {df_model['target'].mean():.2%}"
    )
    return df_model, None
