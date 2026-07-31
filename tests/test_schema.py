from pathlib import Path

import pandas as pd
import pytest

from src.schema import SchemaError, load_canonical_dataset, stable_email_id


MAPPING = {
    "columns": {"subject": "subject", "body": "body", "label": "label"},
    "labels": {"1": "phishing", "0": "legitimate"},
}


def test_stable_email_id_is_deterministic():
    first = stable_email_id(1, "subject", "body", 1)
    second = stable_email_id(1, "subject", "body", 1)
    assert first == second
    assert first.startswith("nazario5_000001_")


def test_unknown_label_fails_instead_of_being_guessed(tmp_path: Path):
    path = tmp_path / "raw.csv"
    pd.DataFrame([{"subject": "a", "body": "b", "label": 7}]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="Unmapped label"):
        load_canonical_dataset(path, MAPPING, "utf-8", expected_rows=1)


def test_expected_row_count_is_enforced(tmp_path: Path):
    path = tmp_path / "raw.csv"
    pd.DataFrame([{"subject": "a", "body": "b", "label": 1}]).to_csv(path, index=False)
    with pytest.raises(SchemaError, match="Expected 2 raw rows"):
        load_canonical_dataset(path, MAPPING, "utf-8", expected_rows=2)

