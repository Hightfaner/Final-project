from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import FEATURE_ORDER, model_matrix
from src.io_utils import file_hash, load_json


FINAL_MATRIX_COLUMNS = [
    "email_id",
    "label_original",
    "label_text",
    "template_group_id",
    "split",
    *FEATURE_ORDER,
    "feature_contract_version",
    "sanitisation_version",
]


class ValidationError(RuntimeError):
    """Raised when a frozen output violates a quality gate."""


def validate_feature_contract(contract: dict[str, Any]) -> None:
    if contract.get("version") != "1.0" or contract.get("status") != "frozen":
        raise ValidationError("Feature contract must be version 1.0 with frozen status")
    if contract.get("text_source") != ["sanitised_subject", "sanitised_body"]:
        raise ValidationError("Feature text source contract has changed")
    if contract.get("feature_order") != FEATURE_ORDER:
        raise ValidationError("Frozen feature names or order have changed")


def validate_feature_values(frame: pd.DataFrame) -> None:
    if list(model_matrix(frame).columns) != FEATURE_ORDER:
        raise ValidationError("Model matrix is not exactly the frozen eight columns")
    integer_columns = [column for column in FEATURE_ORDER if column != "uppercase_letter_ratio"]
    for column in integer_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or numeric.lt(0).any():
            raise ValidationError(f"Feature {column} contains missing or negative values")
        if not numeric.map(lambda value: float(value).is_integer()).all():
            raise ValidationError(f"Feature {column} contains non-integer values")
    ratio = pd.to_numeric(frame["uppercase_letter_ratio"], errors="coerce")
    if ratio.isna().any() or not ratio.between(0.0, 1.0, inclusive="both").all():
        raise ValidationError("uppercase_letter_ratio is outside [0, 1]")
    if (frame["ip_address_url_count"] > frame["url_count"]).any():
        raise ValidationError("ip_address_url_count exceeds url_count")


def validate_sanitised_content(frame: pd.DataFrame) -> None:
    required = {"sanitised_subject", "sanitised_body", "processing_status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValidationError(f"Sanitised output is missing columns: {missing}")
    combined = frame["sanitised_subject"].fillna("") + "\n" + frame["sanitised_body"].fillna("")
    active = combined.str.contains(
        r"(?i)<\s*(?:script|style|iframe|object|form)\b", regex=True
    )
    without_static_markers = combined.str.replace(
        r"(?i)\[URL:\s*[^\]\r\n]+\]", "", regex=True
    )
    clickable = without_static_markers.str.contains(
        r"(?i)\b(?:https?://|ftp://|www\.[A-Za-z0-9])", regex=True
    )
    if active.any():
        raise ValidationError("Active HTML remains in sanitised output")
    if clickable.any():
        raise ValidationError("Clickable/raw URLs remain in sanitised output")


def _read_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["email_id"]:
        raise ValidationError(f"ID file must contain only email_id: {path}")
    if frame["email_id"].duplicated().any():
        raise ValidationError(f"Duplicate IDs found in {path}")
    return frame["email_id"].astype(str).tolist()


def validate_output_bundle(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    output_root = root / str(config["output_root"])
    dataset_path = root / str(config["dataset_path"])
    reports_root = root / str(config.get("reports_root", "reports"))
    paths = {
        "sanitised": output_root / "interim" / "nazario_5_sanitised.csv",
        "duplicates": output_root / "interim" / "duplicate_groups.csv",
        "templates": output_root / "interim" / "template_groups.csv",
        "exclusions": output_root / "interim" / "exclusions.csv",
        "train_ids": output_root / "splits" / "train_ids.csv",
        "validation_ids": output_root / "splits" / "validation_ids.csv",
        "test_ids": output_root / "splits" / "test_ids.csv",
        "manifest": output_root / "splits" / "split_manifest.json",
        "all": output_root / "processed" / "feature_matrix_all.csv",
        "train": output_root / "processed" / "feature_matrix_train.csv",
        "validation": output_root / "processed" / "feature_matrix_validation.csv",
        "test": output_root / "processed" / "feature_matrix_test.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise ValidationError(f"Required final outputs are missing: {missing_files}")

    manifest = load_json(paths["manifest"])
    raw_hash = file_hash(dataset_path)
    if raw_hash != manifest.get("dataset_sha256"):
        raise ValidationError("Raw dataset SHA-256 differs from the frozen manifest")
    if bool(dataset_path.stat().st_mode & stat.S_IWRITE):
        raise ValidationError("Raw dataset is not read-only")

    sanitised = pd.read_csv(paths["sanitised"], keep_default_na=False)
    validate_sanitised_content(sanitised)
    expected_raw = int(config["dataset"]["expected_raw_rows"])
    expected_eligible = int(config["dataset"]["expected_eligible_rows"])
    if len(sanitised) != expected_raw:
        raise ValidationError(f"Sanitised row count is {len(sanitised)}, expected {expected_raw}")
    eligible_ids = set(
        sanitised.loc[sanitised["processing_status"].eq("eligible"), "email_id"].astype(str)
    )
    if len(eligible_ids) != expected_eligible:
        raise ValidationError(f"Eligible row count is {len(eligible_ids)}, expected {expected_eligible}")
    exclusions = pd.read_csv(paths["exclusions"], keep_default_na=False)
    if len(exclusions) != expected_raw - expected_eligible:
        raise ValidationError("Exclusion count does not reconcile with raw and eligible counts")

    split_ids = {
        "train": _read_ids(paths["train_ids"]),
        "validation": _read_ids(paths["validation_ids"]),
        "test": _read_ids(paths["test_ids"]),
    }
    split_sets = {name: set(values) for name, values in split_ids.items()}
    if split_sets["train"] & split_sets["validation"]:
        raise ValidationError("Train and validation IDs overlap")
    if split_sets["train"] & split_sets["test"]:
        raise ValidationError("Train and test IDs overlap")
    if split_sets["validation"] & split_sets["test"]:
        raise ValidationError("Validation and test IDs overlap")
    if set().union(*split_sets.values()) != eligible_ids:
        raise ValidationError("Split ID union does not equal the eligible ID set")
    expected_counts = config["split"]["expected_counts"]
    for name in ("train", "validation", "test"):
        if len(split_ids[name]) != int(expected_counts[name]):
            raise ValidationError(f"{name} ID count differs from the frozen target")

    templates = pd.read_csv(paths["templates"])
    template_splits = templates.merge(
        pd.concat(
            [pd.DataFrame({"email_id": values, "split": name}) for name, values in split_ids.items()],
            ignore_index=True,
        ),
        on="email_id",
        how="left",
        validate="one_to_one",
    )
    cross_split = int(template_splits.groupby("template_group_id")["split"].nunique().gt(1).sum())
    if cross_split:
        raise ValidationError(f"Found {cross_split} template groups spanning splits")

    matrices = {name: pd.read_csv(paths[name]) for name in ("all", "train", "validation", "test")}
    if list(matrices["all"].columns) != FINAL_MATRIX_COLUMNS:
        raise ValidationError("Final all-matrix schema or column order has changed")
    if len(matrices["all"]) != expected_eligible:
        raise ValidationError("Final all-matrix row count is incorrect")
    validate_feature_values(matrices["all"])
    for name in ("train", "validation", "test"):
        if list(matrices[name].columns) != FINAL_MATRIX_COLUMNS:
            raise ValidationError(f"Final {name} matrix schema or column order has changed")
        if set(matrices[name]["email_id"].astype(str)) != split_sets[name]:
            raise ValidationError(f"Final {name} matrix does not exactly match its frozen IDs")

    keyword_columns = [
        "urgency_word_count",
        "credential_word_count",
        "action_word_count",
        "money_related_word_count",
    ]
    zero_keyword_columns = [column for column in keyword_columns if not matrices["all"][column].gt(0).any()]
    if zero_keyword_columns:
        raise ValidationError(f"Frozen keyword features are still all zero: {zero_keyword_columns}")

    checksums_path = reports_root / "output_checksums.json"
    if checksums_path.is_file():
        checksums = load_json(checksums_path)
        for relative_path, expected_hash in checksums.get("sha256", {}).items():
            candidate = root / relative_path
            if not candidate.is_file() or file_hash(candidate) != expected_hash:
                raise ValidationError(f"Checksum mismatch: {relative_path}")

    return {
        "status": "PASS",
        "dataset_sha256": raw_hash,
        "raw_rows": len(sanitised),
        "eligible_rows": len(eligible_ids),
        "excluded_rows": len(exclusions),
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "cross_split_template_group_violations": cross_split,
        "matrix_shapes": {name: list(frame.shape) for name, frame in matrices.items()},
        "keyword_nonzero_rows": {
            column: int(matrices["all"][column].gt(0).sum()) for column in keyword_columns
        },
    }
