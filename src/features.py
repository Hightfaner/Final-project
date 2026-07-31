from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from src.sanitiser import URL_MARKER_RE


FEATURE_ORDER = [
    "url_count",
    "ip_address_url_count",
    "urgency_word_count",
    "credential_word_count",
    "action_word_count",
    "money_related_word_count",
    "uppercase_letter_ratio",
    "exclamation_mark_count",
]
KEYWORD_CATEGORIES = ("urgency", "credential", "action", "money_related")


class KeywordConfigError(ValueError):
    """Raised when the frozen keyword contract is incomplete or mutable."""


@lru_cache(maxsize=None)
def _whole_term_pattern(term: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in term.split()]
    body = r"\s+".join(pieces)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class KeywordMatcher:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        status = self.config.get("keywords_status", self.config.get("status"))
        version = self.config.get("keywords_version", self.config.get("version"))
        if status != "frozen":
            raise KeywordConfigError(f"Final feature extraction requires frozen status, found {status!r}")
        if version != "1.0":
            raise KeywordConfigError(f"Expected keyword version 1.0, found {version!r}")
        for category in KEYWORD_CATEGORIES:
            values = self.config.get(category)
            if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values):
                raise KeywordConfigError(f"Keyword category {category!r} must be a non-empty string list")

    def normalise_variants(self, text: str) -> str:
        result = text.lower()
        mappings = self.config.get("variant_normalisation", {})
        for source, target in sorted(mappings.items(), key=lambda item: (-len(item[0]), item[0])):
            result = _whole_term_pattern(str(source)).sub(str(target), result)
        return result

    def count_category(self, text: str, category: str) -> int:
        if category not in KEYWORD_CATEGORIES:
            raise KeyError(category)
        working = self.normalise_variants(text)
        terms = sorted(
            self.config[category],
            key=lambda value: (-len(value.split()), -len(value), value),
        )
        phrases = [term for term in terms if " " in term]
        words = [term for term in terms if " " not in term]
        count = 0
        for term in phrases + words:
            pattern = _whole_term_pattern(term)
            matches = list(pattern.finditer(working))
            count += len(matches)
            if matches:
                chars = list(working)
                for match in matches:
                    chars[match.start() : match.end()] = " " * (match.end() - match.start())
                working = "".join(chars)
        return count


def _ip_url_count(text: str) -> int:
    count = 0
    for marker in URL_MARKER_RE.findall(text):
        host = marker.strip().split("/", 1)[0]
        try:
            ipaddress.ip_address(host)
        except ValueError:
            continue
        count += 1
    return count


def extract_features(subject: object, body: object, matcher: KeywordMatcher) -> dict[str, int | float]:
    subject_text = "" if subject is None else str(subject)
    body_text = "" if body is None else str(body)
    text = f"{subject_text}\n{body_text}"
    latin_letters = re.findall(r"[A-Za-z]", text)
    uppercase = sum(1 for character in latin_letters if "A" <= character <= "Z")
    ratio = uppercase / len(latin_letters) if latin_letters else 0.0
    markers = URL_MARKER_RE.findall(text)
    return {
        "url_count": len(markers),
        "ip_address_url_count": _ip_url_count(text),
        "urgency_word_count": matcher.count_category(text, "urgency"),
        "credential_word_count": matcher.count_category(text, "credential"),
        "action_word_count": matcher.count_category(text, "action"),
        "money_related_word_count": matcher.count_category(text, "money_related"),
        "uppercase_letter_ratio": ratio,
        "exclamation_mark_count": text.count("!"),
    }


def feature_frame(frame: pd.DataFrame, matcher: KeywordMatcher) -> pd.DataFrame:
    rows = [
        extract_features(subject, body, matcher)
        for subject, body in zip(
            frame["sanitised_subject"], frame["sanitised_body"], strict=True
        )
    ]
    result = pd.DataFrame(rows, index=frame.index)
    return result[FEATURE_ORDER]


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURE_ORDER) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature matrix is missing frozen columns: {missing}")
    return frame.loc[:, FEATURE_ORDER].copy()
