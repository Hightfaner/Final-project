from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.schema import DatasetSnapshot


def _blank_count(series: pd.Series) -> int:
    return int(series.fillna("").astype(str).str.strip().eq("").sum())


def build_dataset_audit(frame: pd.DataFrame, snapshot: DatasetSnapshot) -> dict[str, Any]:
    body = frame["raw_body"].fillna("").astype(str)
    subject = frame["raw_subject"].fillna("").astype(str)
    labels = frame["label_text"].value_counts().sort_index()
    combined = subject + "\n" + body
    lengths = body.str.len()
    return {
        "file": {
            "path": snapshot.path,
            "size_bytes": snapshot.size_bytes,
            "encoding": snapshot.encoding,
            "row_count": snapshot.row_count,
            "columns": list(snapshot.columns),
            "md5": snapshot.md5,
            "sha256": snapshot.sha256,
        },
        "labels": {
            "counts": {str(key): int(value) for key, value in labels.items()},
            "invalid_count": 0,
        },
        "text": {
            "blank_subject_count": _blank_count(subject),
            "blank_body_count": _blank_count(body),
            "body_length_min": int(lengths.min()),
            "body_length_max": int(lengths.max()),
            "body_length_mean": float(lengths.mean()),
            "html_like_count": int(combined.str.contains(r"<\s*[A-Za-z][^>]*>", regex=True).sum()),
        },
        "safety": {
            "url_like_count": int(combined.str.contains(r"(?i)(?:https?://|www\.)", regex=True).sum()),
            "active_tag_count": int(
                combined.str.contains(r"(?i)<\s*(?:script|style|iframe|object|form)\b", regex=True).sum()
            ),
            "email_like_count": int(
                combined.str.contains(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", regex=True).sum()
            ),
        },
        "safe_samples": {
            "blank_subject_email_ids": frame.loc[subject.str.strip().eq(""), "email_id"].head(10).tolist(),
            "blank_body_email_ids": frame.loc[body.str.strip().eq(""), "email_id"].head(10).tolist(),
        },
    }


def audit_markdown(audit: dict[str, Any]) -> str:
    file_info = audit["file"]
    labels = audit["labels"]["counts"]
    text = audit["text"]
    safety = audit["safety"]
    return "\n".join(
        [
            "# Dataset Audit",
            "",
            "## File contract",
            "",
            f"- Rows: {file_info['row_count']}",
            f"- Columns: {', '.join(file_info['columns'])}",
            f"- Encoding: {file_info['encoding']}",
            f"- MD5: `{file_info['md5']}`",
            f"- SHA-256: `{file_info['sha256']}`",
            "",
            "## Labels and text quality",
            "",
            f"- Label counts: {labels}",
            f"- Blank subjects: {text['blank_subject_count']}",
            f"- Blank bodies before sanitisation: {text['blank_body_count']}",
            f"- Body length range: {text['body_length_min']}–{text['body_length_max']}",
            f"- HTML-like records: {text['html_like_count']}",
            "",
            "## Safety indicators",
            "",
            f"- URL-like records: {safety['url_like_count']}",
            f"- Active-tag records: {safety['active_tag_count']}",
            f"- Email-like records: {safety['email_like_count']}",
            "",
            "Only counts and email IDs are reported; raw email bodies are not reproduced.",
            "",
        ]
    )

