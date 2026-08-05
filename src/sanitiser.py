from __future__ import annotations

import html
import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from bs4 import BeautifulSoup


URL_MARKER_RE = re.compile(r"\[URL:\s*([^\]\r\n]+)\]", re.IGNORECASE)
STATIC_MARKER_RE = re.compile(r"\[(?:URL|LINK):\s*[^\]\r\n]+\]", re.IGNORECASE)
LEGACY_URL_WRAPPER_RE = re.compile(
    r"(?i)<URL:\s*((?:https?://|www\.)[^>\s]+)\s*>"
)
RAW_LINK_RE = re.compile(
    r"(?i)(?:(?:https?://|ftp://)[^\s<>\"'\]]+|(?<![A-Za-z0-9])www\.[^\s<>\"'\]]+)"
)
EMAIL_RE = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._%+\-]{0,63})@"
    r"([A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,63})\b"
)
PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .\-]?)?(?:\(?\d{2,4}\)?[ .\-]?){2,4}\d{2,4}(?!\w)"
)
ACTIVE_TAGS = ("script", "style", "iframe", "object", "form")
HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(?:html|body|head|title|p|div|span|a|img|br|table|thead|tbody|tfoot|tr|td|th|ul|ol|li|h[1-6]|script|style|iframe|object|form|font|meta|link|source|video|audio|embed|blockquote|pre|hr)\b[^>]*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SanitisationResult:
    text: str


def _decode_entities(value: str) -> str:
    current = value
    for _ in range(8):
        decoded = html.unescape(current)
        if decoded == current:
            return current
        current = decoded
    return current


def _redact_phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[PHONE]" if 7 <= len(digits) <= 15 else match.group(0)


def _normalise_link_target(value: str) -> str:
    candidate = value.strip().rstrip(".,;:!?)}>")
    parse_value = (
        candidate
        if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", candidate)
        else f"http://{candidate}"
    )
    parsed = urlsplit(parse_value)
    host = (parsed.hostname or "unknown-host").lower()
    try:
        host = str(ipaddress.ip_address(host))
    except ValueError:
        pass
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    suffix = path + (f"?{parsed.query}" if parsed.query else "")
    suffix = EMAIL_RE.sub(lambda match: f"[USER]@{match.group(2).lower()}", suffix)
    suffix = PHONE_RE.sub(_redact_phone, suffix)
    suffix = re.sub(
        r"(?i)(https?|ftp)://",
        lambda match: f"{match.group(1).lower()}-colon//",
        suffix,
    )
    return re.sub(r"[\x00-\x1f\x7f\[\]]", "", f"{host}{suffix}")[:240]


def _is_http_link(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered.startswith(("http://", "https://", "www."))


def _replace_raw_links(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,;:!?)}>":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        marker = "URL" if _is_http_link(raw) else "LINK"
        return f"[{marker}: {_normalise_link_target(raw)}]{trailing}"

    return RAW_LINK_RE.sub(replace, text)


def _html_to_text(value: str) -> str:
    if not HTML_TAG_RE.search(value):
        return value
    soup = BeautifulSoup(value, "lxml")
    for tag_name in ACTIVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all("img"):
        tag.replace_with(" [REMOTE IMAGE REMOVED] ")
    for tag in soup.find_all("a"):
        href = str(tag.get("href") or "")
        anchor_text = tag.get_text(" ", strip=True)
        if re.match(r"(?i)^(?:https?://|www\.)", href):
            target = f"[URL: {_normalise_link_target(href)}]"
            tag.replace_with(f"{anchor_text} {target}".strip())
        elif re.match(r"(?i)^ftp://", href):
            target = f"[LINK: {_normalise_link_target(href)}]"
            tag.replace_with(f"{anchor_text} {target}".strip())
        else:
            tag.replace_with(anchor_text)
    for tag_name in ("link", "source", "video", "audio", "embed"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    return soup.get_text("\n")


def _normalise_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in value.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def sanitise_text(value: object) -> SanitisationResult:
    source = "" if value is None else str(value)
    text = _html_to_text(_decode_entities(source))
    text = LEGACY_URL_WRAPPER_RE.sub(
        lambda match: f"[URL: {_normalise_link_target(match.group(1))}]", text
    )

    markers: list[str] = []

    def protect_marker(match: re.Match[str]) -> str:
        token = f"__STATIC_LINK_MARKER_{len(markers)}__"
        markers.append(match.group(0))
        return token

    # Protect markers already produced by an earlier run before looking for
    # raw links, then protect the new markers before PII replacement.
    text = STATIC_MARKER_RE.sub(protect_marker, text)
    text = _replace_raw_links(text)
    text = STATIC_MARKER_RE.sub(protect_marker, text)
    text = _normalise_whitespace(text)
    text = EMAIL_RE.sub(lambda match: f"[USER]@{match.group(2).lower()}", text)
    for _ in range(64):
        updated = PHONE_RE.sub(_redact_phone, text)
        if updated == text:
            break
        text = updated
    else:
        raise ValueError("Phone redaction did not reach a fixed point")
    for index, marker in enumerate(markers):
        text = text.replace(f"__STATIC_LINK_MARKER_{index}__", marker)
    return SanitisationResult(text=_normalise_whitespace(text))
