import pandas as pd
import pytest

from src.splitter import build_fixed_split
from src.template_grouping import TemplateGroupingError, build_groups


def _grouping_frame():
    return pd.DataFrame(
        [
            {
                "email_id": "a",
                "raw_subject": "Hello Alice",
                "raw_body": "Call 1234567",
                "sanitised_subject": "Hello [USER]",
                "sanitised_body": "Call [PHONE]",
                "label_text": "phishing",
                "label_binary": 1,
            },
            {
                "email_id": "b",
                "raw_subject": "Hello Bob",
                "raw_body": "Call 7654321",
                "sanitised_subject": "Hello [USER]",
                "sanitised_body": "Call [PHONE]",
                "label_text": "legitimate",
                "label_binary": 0,
            },
            {
                "email_id": "c",
                "raw_subject": "Other",
                "raw_body": "Different",
                "sanitised_subject": "Other",
                "sanitised_body": "Different",
                "label_text": "phishing",
                "label_binary": 1,
            },
        ]
    )


def test_raw_and_sanitised_duplicates_are_distinguished():
    result = build_groups(_grouping_frame(), {"near_similarity_enabled": False})
    assert result.summary["raw_exact_duplicate_groups"] == 0
    assert result.summary["sanitised_exact_duplicate_groups"] == 1
    assert result.summary["sanitised_exact_duplicate_records"] == 2


def test_mixed_label_sanitised_duplicate_is_flagged_not_deleted():
    result = build_groups(_grouping_frame(), {"near_similarity_enabled": False})
    assert len(result.frame) == 3
    assert result.summary["mixed_label_sanitised_duplicate_groups"] == 1
    duplicates = result.frame.loc[result.frame["duplicate_group_size"].eq(2)]
    assert not duplicates["duplicate_label_consistent"].any()


def test_grouping_does_not_use_labels():
    first = build_groups(_grouping_frame(), {"near_similarity_enabled": False}).frame
    changed = _grouping_frame().copy()
    changed["label_text"] = changed["label_text"].iloc[::-1].to_numpy()
    second = build_groups(changed, {"near_similarity_enabled": False}).frame
    assert first.set_index("email_id")["template_group_id"].to_dict() == second.set_index(
        "email_id"
    )["template_group_id"].to_dict()


def test_oversized_template_group_stops_for_review():
    with pytest.raises(TemplateGroupingError, match="review limit"):
        build_groups(
            _grouping_frame(),
            {"near_similarity_enabled": False, "max_group_size": 1},
        )


def test_split_is_exact_group_safe_and_deterministic():
    rows = []
    for index in range(20):
        rows.append(
            {
                "email_id": f"e{index:02d}",
                "template_group_id": "paired" if index < 2 else f"single_{index:02d}",
                "label_binary": index % 2,
            }
        )
    frame = pd.DataFrame(rows)
    config = {
        "expected_counts": {"train": 14, "validation": 3, "test": 3},
        "seed": 42,
        "max_class_ratio_deviation": 0.20,
    }
    first = build_fixed_split(frame, config)
    second = build_fixed_split(frame, config)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert first.summary["actual_counts"] == {"train": 14, "validation": 3, "test": 3}
    assert first.summary["cross_split_template_group_violations"] == 0

