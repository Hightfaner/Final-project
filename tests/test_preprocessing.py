from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.features import KeywordMatcher, extract_features
from src.io_utils import load_yaml
from src.pipeline import (
    BASE_OUTPUT_COLUMNS,
    PreprocessingError,
    _new_stratified_split,
    _validate_manual_review,
    _validate_output,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "configs" / "pipeline.yaml")
FEATURES = CONFIG["feature_order"]


def valid_frame() -> pd.DataFrame:
    splits = ["train", "train", "validation", "validation", "test", "test", "train", "test"]
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    rows = []
    for index, (split, label) in enumerate(zip(splits, labels, strict=True)):
        row = {
            "email_id": f"email_{index}",
            "sanitized_subject": "Subject",
            "sanitized_body": "Body",
            "label": label,
            "split": split,
            **{name: 0 for name in FEATURES},
        }
        row["uppercase_letter_ratio"] = 0.25
        rows.append(row)
    return pd.DataFrame(rows)[BASE_OUTPUT_COLUMNS + FEATURES]


def test_01_required_columns_and_binary_labels() -> None:
    checks = _validate_output(valid_frame(), FEATURES)
    assert checks["必需列存在且标签仅为0/1"] == "PASS"


def test_02_features_are_finite_numeric() -> None:
    broken = valid_frame()
    broken.loc[0, "url_count"] = float("nan")
    with pytest.raises(PreprocessingError, match="NaN or Inf"):
        _validate_output(broken, FEATURES)


def test_03_feature_ranges_are_enforced() -> None:
    broken_ratio = valid_frame()
    broken_ratio.loc[0, "uppercase_letter_ratio"] = 1.1
    with pytest.raises(PreprocessingError, match="outside"):
        _validate_output(broken_ratio, FEATURES)
    broken_count = valid_frame()
    broken_count.loc[0, "url_count"] = -1
    with pytest.raises(PreprocessingError, match="negative"):
        _validate_output(broken_count, FEATURES)


def test_04_email_ids_are_unique_and_splits_disjoint() -> None:
    broken = valid_frame()
    broken.loc[1, "email_id"] = broken.loc[0, "email_id"]
    with pytest.raises(PreprocessingError, match="uniqueness"):
        _validate_output(broken, FEATURES)


def test_05_each_split_contains_both_classes() -> None:
    broken = valid_frame()
    broken.loc[broken["split"].eq("validation"), "label"] = 0
    with pytest.raises(PreprocessingError, match="both labels"):
        _validate_output(broken, FEATURES)


def test_06_seed_and_feature_output_are_deterministic() -> None:
    split_input = pd.DataFrame(
        {"email_id": [f"id_{index:02d}" for index in range(40)], "label": [index % 2 for index in range(40)]}
    )
    assert _new_stratified_split(split_input, CONFIG["split"]["ratios"], 42) == _new_stratified_split(
        split_input, CONFIG["split"]["ratios"], 42
    )
    matcher = KeywordMatcher(CONFIG["keywords"])
    assert extract_features("URGENT!", "Visit https://192.0.2.1/login", matcher) == extract_features(
        "URGENT!", "Visit https://192.0.2.1/login", matcher
    )


def test_07_eight_manual_review_ids_are_recorded() -> None:
    frame = valid_frame()
    review = {
        "status": "approved",
        "reviewed_date": "2026-08-03",
        "email_ids": frame["email_id"].tolist(),
    }
    assert len(_validate_manual_review(frame, {"manual_spot_check": review})) == 8
    pending = deepcopy(review)
    pending["status"] = "pending"
    with pytest.raises(PreprocessingError, match="not been approved"):
        _validate_manual_review(frame, {"manual_spot_check": pending})
