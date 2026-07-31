from copy import deepcopy

import pandas as pd
import pytest

from src.features import FEATURE_ORDER
from src.validation import (
    ValidationError,
    validate_feature_contract,
    validate_feature_values,
    validate_sanitised_content,
)


def test_feature_contract_accepts_frozen_exact_order(feature_contract):
    validate_feature_contract(feature_contract)


def test_feature_contract_rejects_ninth_feature(feature_contract):
    broken = deepcopy(feature_contract)
    broken["feature_order"] = [*FEATURE_ORDER, "forbidden_ninth_feature"]
    with pytest.raises(ValidationError, match="order"):
        validate_feature_contract(broken)


def test_feature_value_ranges_are_enforced():
    row = {column: 0 for column in FEATURE_ORDER}
    row["url_count"] = 1
    row["ip_address_url_count"] = 2
    with pytest.raises(ValidationError, match="exceeds"):
        validate_feature_values(pd.DataFrame([row]))


def test_sanitised_content_rejects_raw_clickable_url():
    frame = pd.DataFrame(
        [
            {
                "sanitised_subject": "safe",
                "sanitised_body": "https://unsafe.example",
                "processing_status": "eligible",
            }
        ]
    )
    with pytest.raises(ValidationError, match="Clickable"):
        validate_sanitised_content(frame)


def test_sanitised_content_accepts_static_url_marker_with_www_host():
    frame = pd.DataFrame(
        [
            {
                "sanitised_subject": "safe",
                "sanitised_body": "[URL: www.example.com/path]",
                "processing_status": "eligible",
            }
        ]
    )
    validate_sanitised_content(frame)


def test_sanitised_content_does_not_treat_prose_www_prefix_as_url():
    frame = pd.DataFrame(
        [
            {
                "sanitised_subject": "safe",
                "sanitised_body": "The phrase with www. prefix tests another branch.",
                "processing_status": "eligible",
            }
        ]
    )
    validate_sanitised_content(frame)
