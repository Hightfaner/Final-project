from __future__ import annotations

import html
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from bs4 import BeautifulSoup


URL_MARKER_RE = re.compile(r"\[URL:\s*([^\]\r\n]+)\]", re.IGNORECASE)
RAW_URL_RE = re.compile(r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>\"'\]]+")
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])([A-Za-z0-9][A-Za-z0-9._%+\-]{0,63})@"
    r"([A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,63})\b"
)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .\-]?)?(?:\(?\d{2,4}\)?[ .\-]?){2,4}\d{2,4}(?!\w)"
)
ATTACHMENT_RE = re.compile(
    r"(?im)^\s*(?:content-disposition\s*:\s*attachment[^\r\n]*|attachment\s*:\s*[^\r\n]+)\s*$"
)
ACTIVE_TAGS = ("script", "style", "iframe", "object", "form")
HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?:html|body|head|title|p|div|span|a|img|br|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|h[1-6]|script|style|iframe|object|form|font|meta|link|source|video|audio|embed|blockquote|pre|hr)\b[^>]*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SanitisationResult:
    text: str
    url_replacement_count: int
    pii_replacement_count: int
    removed_active_content_count: int


def _decode_entities_fixed_point(value: str) -> str:
    current = value
    for _ in range(8):
        decoded = html.unescape(current)
        if decoded == current:
            break
        current = decoded
    return current


def _normalise_url(value: str) -> str:
    candidate = value.strip().rstrip(".,;:!?)}>")
    parse_value = candidate if "://" in candidate else f"http://{candidate}"
    parsed = urlsplit(parse_value)
    host = (parsed.hostname or "unknown-host").lower()
    try:
        host = str(ipaddress.ip_address(host))
    except ValueError:
        pass
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    safe = re.sub(r"[\x00-\x1f\x7f\[\]]", "", f"{host}{path}")
    return safe[:240]


def _replace_raw_urls(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,;:!?)}>":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        count += 1
        return f"[URL: {_normalise_url(raw)}]{trailing}"

    return RAW_URL_RE.sub(replace, text), count


def _html_to_text(value: str) -> tuple[str, int, int]:
    if not HTML_TAG_RE.search(value):
        return value, 0, 0
    soup = BeautifulSoup(value, "lxml")
    removed = 0
    url_count = 0
    for tag_name in ACTIVE_TAGS:
        for tag in soup.find_all(tag_name):
            removed += 1
            tag.decompose()
    for tag in soup.find_all("img"):
        source = str(tag.get("src") or "")
        if source:
            tag.replace_with(" [REMOTE IMAGE REMOVED] ")
        else:
            tag.decompose()
        removed += 1
    for tag in soup.find_all("a"):
        href = str(tag.get("href") or "")
        anchor_text = tag.get_text(" ", strip=True)
        if re.match(r"(?i)^(?:https?://|ftp://|www\.)", href):
            marker = f"[URL: {_normalise_url(href)}]"
            tag.replace_with(f"{anchor_text} {marker}".strip())
            url_count += 1
        else:
            tag.replace_with(anchor_text)
    for tag_name in ("link", "source", "video", "audio", "embed"):
        for tag in soup.find_all(tag_name):
            removed += 1
            tag.decompose()
    return soup.get_text("\n"), url_count, removed


def _normalise_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def sanitise_text(value: object) -> SanitisationResult:
    source = "" if value is None else str(value)
    source = _decode_entities_fixed_point(source)
    text, html_url_count, removed_count = _html_to_text(source)

    attachment_count = len(ATTACHMENT_RE.findall(text))
    if attachment_count:
        text = ATTACHMENT_RE.sub("[ATTACHMENT REMOVED]", text)

    protected_markers: list[str] = []

    def protect_marker(match: re.Match[str]) -> str:
        token = f"__STATIC_URL_MARKER_{len(protected_markers)}__"
        protected_markers.append(match.group(0))
        return token

    text = URL_MARKER_RE.sub(protect_marker, text)
    text, raw_url_count = _replace_raw_urls(text)
    for index, marker in enumerate(protected_markers):
        text = text.replace(f"__STATIC_URL_MARKER_{index}__", marker)
    text = _normalise_whitespace(text)
    pii_count = 0

    def replace_email(match: re.Match[str]) -> str:
        nonlocal pii_count
        pii_count += 1
        return f"[USER]@{match.group(2).lower()}"

    text = EMAIL_RE.sub(replace_email, text)

    def replace_phone(match: re.Match[str]) -> str:
        nonlocal pii_count
        candidate = match.group(0)
        digits = re.sub(r"\D", "", candidate)
        if not 7 <= len(digits) <= 15:
            return candidate
        pii_count += 1
        return "[PHONE]"

    for _ in range(8):
        updated = PHONE_RE.sub(replace_phone, text)
        if updated == text:
            break
        text = updated
    text = _normalise_whitespace(text)
    return SanitisationResult(
        text=text,
        url_replacement_count=html_url_count + raw_url_count,
        pii_replacement_count=pii_count,
        removed_active_content_count=removed_count + attachment_count,
    )


def meaningful_body(text: str) -> bool:
    # The frozen exclusion rule is "completely missing or blank body". Do not
    # impose an ASCII-language or semantic-content test: non-Latin text,
    # punctuation-only source records, and safe evidence markers are not blank.
    return bool(text.strip())
