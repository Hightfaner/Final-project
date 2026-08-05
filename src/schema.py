from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when the source CSV does not meet the minimal input contract."""


def _label_key(value: Any) -> str:
    if pd.isna(value):
        raise SchemaError("Missing labels are not permitted")
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def stable_email_id(
    original_row_number: int,
    raw_subject: str,
    raw_body: str,
    label: Any,
) -> str:
    payload = f"{raw_subject}\n{raw_body}\n{_label_key(label)}"
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"nazario5_{original_row_number:06d}_{content_hash}"


def load_canonical_dataset(dataset_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    columns = config.get("columns", {})
    required_roles = {"subject", "body", "label"}
    if not required_roles.issubset(columns):
        raise SchemaError(f"Missing column mappings: {sorted(required_roles - set(columns))}")

    raw = pd.read_csv(dataset_path, encoding=str(config["encoding"]), low_memory=False)
    expected_rows = int(config.get("expected_rows", len(raw)))
    if len(raw) != expected_rows:
        raise SchemaError(f"Expected {expected_rows} rows, found {len(raw)}")

    missing_columns = sorted(set(columns.values()) - set(raw.columns))
    if missing_columns:
        raise SchemaError(f"Raw CSV is missing columns: {missing_columns}")

    frame = pd.DataFrame(
        {
            "original_row_number": range(1, len(raw) + 1),
            "raw_subject": raw[columns["subject"]].fillna("").astype(str),
            "raw_body": raw[columns["body"]].fillna("").astype(str),
        }
    )
    label_keys = raw[columns["label"]].map(_label_key)
    label_mapping = {str(key): int(value) for key, value in config.get("labels", {}).items()}
    unknown = sorted(set(label_keys) - set(label_mapping))
    if unknown:
        raise SchemaError(f"Unknown label values: {unknown}")
    frame["label"] = label_keys.map(label_mapping).astype(int)
    frame["email_id"] = [
        stable_email_id(row_number, subject, body, label)
        for row_number, subject, body, label in zip(
            frame["original_row_number"],
            frame["raw_subject"],
            frame["raw_body"],
            frame["label"],
            strict=True,
        )
    ]
    if not frame["email_id"].is_unique:
        raise SchemaError("Generated email_id values are not unique")
    return frame
