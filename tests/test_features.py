from copy import deepcopy

import pandas as pd
import pytest

from src.features import (
    FEATURE_ORDER,
    KeywordConfigError,
    KeywordMatcher,
    extract_features,
    feature_frame,
    model_matrix,
)


def test_k01_keyword_matching_is_case_insensitive(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features("URGENT urgent Urgent", "body", matcher)
    assert result["urgency_word_count"] == 3


def test_k02_whole_word_boundary_prevents_pin_substrings(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features("shopping mapping", "", matcher)
    assert result["credential_word_count"] == 0


def test_k03_multiword_phrase_counts_once(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features("action required", "", matcher)
    assert result["urgency_word_count"] == 1


def test_k04_hyphen_variant_matches_space_variant(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    hyphenated = extract_features("sign-in", "", matcher)
    spaced = extract_features("sign in", "", matcher)
    assert hyphenated["credential_word_count"] == spaced["credential_word_count"] == 1


def test_k05_repeated_occurrences_are_all_counted(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features("verify, then verify", "", matcher)
    assert result["action_word_count"] == 2


def test_k06_same_text_different_labels_has_identical_features(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    frame = pd.DataFrame(
        {
            "sanitised_subject": ["URGENT verify", "URGENT verify"],
            "sanitised_body": ["Pay invoice!", "Pay invoice!"],
            "label_text": ["phishing", "legitimate"],
        }
    )
    features = feature_frame(frame, matcher)
    assert features.iloc[0].to_dict() == features.iloc[1].to_dict()


def test_k07_changing_label_does_not_change_features(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    base = pd.DataFrame(
        {"sanitised_subject": ["Account alert"], "sanitised_body": ["Review payment"]}
    )
    first = feature_frame(base.assign(label_binary=1), matcher)
    second = feature_frame(base.assign(label_binary=0), matcher)
    pd.testing.assert_frame_equal(first, second)


def test_k08_empty_subject_with_nonempty_body_is_supported(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features("", "Please review your account", matcher)
    assert result["credential_word_count"] == 1
    assert result["action_word_count"] == 1


def test_k09_feature_extraction_is_reproducible(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    first = extract_features("ALERT!", "Visit [URL: 192.0.2.1/login]", matcher)
    second = extract_features("ALERT!", "Visit [URL: 192.0.2.1/login]", matcher)
    assert first == second


@pytest.mark.parametrize("mutation", ["status", "empty"])
def test_k10_final_rejects_mutable_or_empty_keyword_config(keyword_config, mutation):
    broken = deepcopy(keyword_config)
    if mutation == "status":
        broken["keywords_status"] = "draft"
    else:
        broken["urgency"] = []
    with pytest.raises(KeywordConfigError):
        KeywordMatcher(broken)


def test_url_ip_uppercase_and_exclamation_features(keyword_config):
    matcher = KeywordMatcher(keyword_config)
    result = extract_features(
        "ALERT!",
        "A [URL: 192.0.2.1/login] B [URL: example.com/]!",
        matcher,
    )
    assert result["url_count"] == 2
    assert result["ip_address_url_count"] == 1
    assert 0 < result["uppercase_letter_ratio"] <= 1
    assert result["exclamation_mark_count"] == 2


def test_model_matrix_returns_exact_frozen_order():
    frame = pd.DataFrame([{column: 0 for column in FEATURE_ORDER}]).assign(email_id="x", label=1)
    assert list(model_matrix(frame).columns) == FEATURE_ORDER

