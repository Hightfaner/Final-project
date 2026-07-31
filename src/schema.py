from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.io_utils import file_hash


class SchemaError(ValueError):
    """Raised when the raw dataset violates the frozen input contract."""


@dataclass(frozen=True)
class DatasetSnapshot:
    path: str
    size_bytes: int
    encoding: str
    row_count: int
    columns: tuple[str, ...]
    md5: str
    sha256: str


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
    label_original: Any,
) -> str:
    payload = f"{raw_subject}\n{raw_body}\n{_label_key(label_original)}"
    content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"nazario5_{original_row_number:06d}_{content_hash}"


def load_canonical_dataset(
    dataset_path: Path,
    mapping_config: dict[str, Any],
    encoding: str,
    expected_rows: int | None = None,
) -> tuple[pd.DataFrame, DatasetSnapshot]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    columns = mapping_config.get("columns", {})
    label_mapping = mapping_config.get("labels", {})
    required_roles = {"subject", "body", "label"}
    if not required_roles.issubset(columns):
        missing = sorted(required_roles - set(columns))
        raise SchemaError(f"Missing required column mappings: {missing}")

    raw = pd.read_csv(dataset_path, encoding=encoding, low_memory=False)
    required_source_columns = set(columns.values())
    absent = sorted(required_source_columns - set(raw.columns))
    if absent:
        raise SchemaError(f"Raw CSV is missing configured columns: {absent}")
    if expected_rows is not None and len(raw) != expected_rows:
        raise SchemaError(f"Expected {expected_rows} raw rows, found {len(raw)}")

    canonical = pd.DataFrame(index=raw.index)
    canonical["original_row_number"] = range(1, len(raw) + 1)
    for role, source_name in columns.items():
        canonical[f"raw_{role}"] = raw[source_name]

    canonical["raw_subject"] = canonical["raw_subject"].fillna("").astype(str)
    canonical["raw_body"] = canonical["raw_body"].fillna("").astype(str)
    label_keys = canonical["raw_label"].map(_label_key)
    unknown = sorted(set(label_keys) - set(label_mapping))
    if unknown:
        raise SchemaError(f"Unmapped label values: {unknown}")

    canonical["label_original"] = label_keys.astype(int)
    canonical["label_text"] = label_keys.map(label_mapping)
    canonical["label_binary"] = canonical["label_original"].astype(int)
    canonical["email_id"] = [
        stable_email_id(row_number, subject, body, label)
        for row_number, subject, body, label in zip(
            canonical["original_row_number"],
            canonical["raw_subject"],
            canonical["raw_body"],
            canonical["label_original"],
            strict=True,
        )
    ]
    if not canonical["email_id"].is_unique:
        raise SchemaError("Generated email_id values are not unique")

    snapshot = DatasetSnapshot(
        path=str(dataset_path),
        size_bytes=dataset_path.stat().st_size,
        encoding=encoding,
        row_count=len(raw),
        columns=tuple(str(column) for column in raw.columns),
        md5=file_hash(dataset_path, "md5"),
        sha256=file_hash(dataset_path, "sha256"),
    )
    return canonical, snapshot

