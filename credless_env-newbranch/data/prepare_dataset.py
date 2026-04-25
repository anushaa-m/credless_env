from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credless_model.dataset_pipeline import TARGET_COLUMN, clean_dataset

DATA_DIR = Path(__file__).resolve().parent
SOURCE_PATH = DATA_DIR / "cd_updated.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "val.csv"
TEST_PATH = DATA_DIR / "test.csv"
DOC_PATH = DATA_DIR / "feature_docs.md"
META_PATH = DATA_DIR / "split_metadata.json"
RANDOM_SEED = 42


def infer_target_semantics(df: pd.DataFrame, target_column: str) -> str:
    checks = {
        "overdraft_count": "lower_is_better",
        "failed_txn_ratio": "lower_is_better",
        "salary_credit_consistency": "higher_is_better",
        "monthlyincome": "higher_is_better",
    }
    score = 0
    for column, rule in checks.items():
        if column not in df.columns:
            continue
        means = df.groupby(target_column)[column].mean().to_dict()
        if 0 not in means or 1 not in means:
            continue
        if rule == "lower_is_better" and means[1] < means[0]:
            score += 1
        if rule == "higher_is_better" and means[1] > means[0]:
            score += 1
    return "1_likely_good_outcome" if score >= 3 else "1_likely_default_or_risk"


def detect_target_column(columns: Iterable[str]) -> str:
    candidates = [TARGET_COLUMN, "fraud", "default", "SeriousDlqin2yrs"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No supported binary target column found. Checked: {candidates}")


def markdown_table(df: pd.DataFrame) -> str:
    header = "| Column | Dtype | Nulls | Unique |"
    sep = "|---|---:|---:|---:|"
    rows = []
    for column in df.columns:
        rows.append(
            f"| `{column}` | `{df[column].dtype}` | {int(df[column].isnull().sum())} | {int(df[column].nunique(dropna=False))} |"
        )
    return "\n".join([header, sep, *rows])


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {SOURCE_PATH}")

    raw = pd.read_csv(SOURCE_PATH)
    target_column = detect_target_column(raw.columns)
    cleaned = clean_dataset(raw)
    target_semantics = infer_target_semantics(cleaned, target_column)

    train_df, temp_df = train_test_split(
        cleaned,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=cleaned[target_column],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df[target_column],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    metadata = {
        "source_path": str(SOURCE_PATH),
        "target_column": target_column,
        "target_semantics": target_semantics,
        "random_seed": RANDOM_SEED,
        "rows_raw": int(len(raw)),
        "rows_cleaned": int(len(cleaned)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "class_balance": {
            "full": cleaned[target_column].value_counts(normalize=True).sort_index().to_dict(),
            "train": train_df[target_column].value_counts(normalize=True).sort_index().to_dict(),
            "val": val_df[target_column].value_counts(normalize=True).sort_index().to_dict(),
            "test": test_df[target_column].value_counts(normalize=True).sort_index().to_dict(),
        },
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    doc = f"""# CredLess Dataset Documentation

## Source

- Dataset: `{SOURCE_PATH.name}`
- Detected target column: `{target_column}`
- Inferred target semantics: `{target_semantics}`
- Raw rows: {len(raw):,}
- Clean rows after duplicate/null filtering: {len(cleaned):,}
- Raw columns: {len(raw.columns)}
- Clean columns used by the model pipeline: {len(cleaned.columns)}

## Split Summary

- Train: {len(train_df):,} rows ({len(train_df) / len(cleaned):.1%})
- Validation: {len(val_df):,} rows ({len(val_df) / len(cleaned):.1%})
- Test: {len(test_df):,} rows ({len(test_df) / len(cleaned):.1%})

## Class Balance

- Full dataset: `{cleaned[target_column].value_counts(normalize=True).sort_index().to_dict()}`
- Train split: `{train_df[target_column].value_counts(normalize=True).sort_index().to_dict()}`
- Validation split: `{val_df[target_column].value_counts(normalize=True).sort_index().to_dict()}`
- Test split: `{test_df[target_column].value_counts(normalize=True).sort_index().to_dict()}`

## Null Counts

All cleaned columns have zero null values after preprocessing.

## Column Inventory

{markdown_table(cleaned)}
"""
    DOC_PATH.write_text(doc, encoding="utf-8")

    print(f"Wrote {TRAIN_PATH}")
    print(f"Wrote {VAL_PATH}")
    print(f"Wrote {TEST_PATH}")
    print(f"Wrote {DOC_PATH}")
    print(f"Wrote {META_PATH}")


if __name__ == "__main__":
    main()
