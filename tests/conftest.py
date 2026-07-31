from pathlib import Path

import pytest

from src.io_utils import load_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def keyword_config():
    return load_json(ROOT / "configs" / "keywords.json")


@pytest.fixture
def feature_contract():
    return load_json(ROOT / "configs" / "feature_contract.json")

