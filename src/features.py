from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pandas as pd

from src.sanitiser import URL_MARKER_RE


KEYWORD_CATEGORIES = ("urgency", "credential", "action", "money_related")


class KeywordConfigError(ValueError):
    """Raised when the frozen keyword definition is missing or mutable."""


@lru_cache(maxsize=None)
def _whole_term_pattern(term: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in term.split()]
    body = r"\s+".join(pieces)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class KeywordMatcher:
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if self.config.get("keywords_status") != "frozen":
            raise KeywordConfigError("Keyword configuration must have frozen status")
        if str(self.config.get("keywords_version")) != "1.0":
            raise KeywordConfigError("Keyword configuration must be version 1.0")
        for category in KEYWORD_CATEGORIES:
            values = self.config.get(category)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value for value in values
            ):
                raise KeywordConfigError(f"Invalid keyword category: {category}")

    def normalise_variants(self, text: str) -> str:
        result = text.lower()
        mappings = self.config.get("variant_normalisation", {})
        for source, target in sorted(mappings.items(), key=lambda item: (-len(item[0]), item[0])):
            result = _whole_term_pattern(str(source)).sub(str(target), result)
        return result

    def count_category(self, text: str, category: str) -> int:
        working = self.normalise_variants(text)
        terms = sorted(
            self.config[category],
            key=lambda value: (-len(value.split()), -len(value), value),
        )
        count = 0
        for term in [value for value in terms if " " in value] + [
            value for value in terms if " " not in value
        ]:
            matches = list(_whole_term_pattern(term).finditer(working))
            count += len(matches)
            if matches:
                characters = list(working)
                for match in matches:
                    characters[match.start() : match.end()] = " " * (match.end() - match.start())
                working = "".join(characters)
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
    text = f"{'' if subject is None else subject}\n{'' if body is None else body}"
    letters = re.findall(r"[A-Za-z]", text)
    uppercase = sum("A" <= character <= "Z" for character in letters)
    markers = URL_MARKER_RE.findall(text)
    return {
        "url_count": len(markers),
        "ip_address_url_count": _ip_url_count(text),
        "urgency_word_count": matcher.count_category(text, "urgency"),
        "credential_word_count": matcher.count_category(text, "credential"),
        "action_word_count": matcher.count_category(text, "action"),
        "money_related_word_count": matcher.count_category(text, "money_related"),
        "uppercase_letter_ratio": uppercase / len(letters) if letters else 0.0,
        "exclamation_mark_count": text.count("!"),
    }


def feature_frame(
    frame: pd.DataFrame,
    matcher: KeywordMatcher,
    feature_order: list[str],
) -> pd.DataFrame:
    rows = [
        extract_features(subject, body, matcher)
        for subject, body in zip(
            frame["sanitized_subject"], frame["sanitized_body"], strict=True
        )
    ]
    result = pd.DataFrame(rows, index=frame.index)
    if len(feature_order) != 8 or set(feature_order) != set(result.columns):
        raise ValueError("feature_order must contain exactly the frozen eight features")
    return result[feature_order]
