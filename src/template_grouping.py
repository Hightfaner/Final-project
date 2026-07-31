from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class TemplateGroupingError(RuntimeError):
    """Raised when deterministic template grouping cannot satisfy its contract."""


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def template_normalise(subject: str, body: str) -> str:
    text = f"{subject}\n{body}".lower()
    text = re.sub(r"\[url:\s*[^\]]+\]", " <url> ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[user\]@[a-z0-9.\-]+", " <email> ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}",
        " <email> ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[phone\]", " <phone> ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{5,}\b", " <number> ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class GroupingResult:
    frame: pd.DataFrame
    duplicate_groups: pd.DataFrame
    template_groups: pd.DataFrame
    similarity_pairs: pd.DataFrame
    summary: dict[str, Any]


def build_groups(frame: pd.DataFrame, config: dict[str, Any]) -> GroupingResult:
    required = {
        "email_id",
        "raw_subject",
        "raw_body",
        "sanitised_subject",
        "sanitised_body",
        "label_text",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TemplateGroupingError(f"Grouping input is missing columns: {missing}")

    work = frame.reset_index(drop=True).copy()
    raw_text = work["raw_subject"].astype(str) + "\n" + work["raw_body"].astype(str)
    sanitised_text = (
        work["sanitised_subject"].astype(str) + "\n" + work["sanitised_body"].astype(str)
    )
    work["raw_text_hash"] = raw_text.map(_sha)
    work["sanitised_text_hash"] = sanitised_text.map(_sha)
    work["duplicate_group_id"] = work["sanitised_text_hash"].map(lambda value: f"dup_{value[:16]}")
    duplicate_sizes = work.groupby("duplicate_group_id")["email_id"].transform("size")
    work["duplicate_group_size"] = duplicate_sizes.astype(int)
    label_counts = work.groupby("duplicate_group_id")["label_text"].transform("nunique")
    work["duplicate_label_consistent"] = label_counts.eq(1)

    normalised = [
        template_normalise(subject, body)
        for subject, body in zip(
            work["sanitised_subject"], work["sanitised_body"], strict=True
        )
    ]
    signature_hashes = [_sha(value) for value in normalised]
    work["template_signature_hash"] = signature_hashes

    union = UnionFind(len(work))
    signature_first: dict[str, int] = {}
    exact_pairs: list[dict[str, Any]] = []
    for index, signature in enumerate(signature_hashes):
        if signature in signature_first:
            first = signature_first[signature]
            union.union(first, index)
            exact_pairs.append(
                {
                    "email_id_a": work.at[first, "email_id"],
                    "email_id_b": work.at[index, "email_id"],
                    "cosine_similarity": 1.0,
                    "evidence_type": "exact_template_signature",
                }
            )
        else:
            signature_first[signature] = index

    near_pairs: list[dict[str, Any]] = []
    if config.get("near_similarity_enabled", True) and len(work) > 1:
        ngram_range = tuple(int(value) for value in config.get("char_ngram_range", [3, 5]))
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            min_df=int(config.get("min_df", 2)),
            max_features=int(config.get("max_features", 100000)),
            dtype=np.float32,
            lowercase=False,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(normalised)
        threshold = float(config.get("cosine_threshold", 0.95))
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
        neighbors.fit(matrix)
        distances, indices = neighbors.radius_neighbors(
            matrix,
            radius=max(0.0, 1.0 - threshold) + 1e-7,
            return_distance=True,
            sort_results=True,
        )
        for left, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
            for distance, right_value in zip(row_distances, row_indices, strict=True):
                right = int(right_value)
                if right <= left or signature_hashes[left] == signature_hashes[right]:
                    continue
                similarity = 1.0 - float(distance)
                if similarity + 1e-7 < threshold:
                    continue
                union.union(left, right)
                near_pairs.append(
                    {
                        "email_id_a": work.at[left, "email_id"],
                        "email_id_b": work.at[right, "email_id"],
                        "cosine_similarity": round(similarity, 8),
                        "evidence_type": "char_3_5gram_tfidf_cosine",
                    }
                )

    members: dict[int, list[int]] = {}
    for index in range(len(work)):
        members.setdefault(union.find(index), []).append(index)
    group_id_by_index: dict[int, str] = {}
    max_group_size = int(config.get("max_group_size", 250))
    for indices_in_group in members.values():
        ids = sorted(str(work.at[index, "email_id"]) for index in indices_in_group)
        group_id = f"tpl_{_sha(chr(10).join(ids))[:16]}"
        if len(indices_in_group) > max_group_size:
            raise TemplateGroupingError(
                f"Template group {group_id} has {len(indices_in_group)} records, "
                f"exceeding configured review limit {max_group_size}"
            )
        for index in indices_in_group:
            group_id_by_index[index] = group_id
    work["template_group_id"] = [group_id_by_index[index] for index in range(len(work))]
    work["template_group_size"] = work.groupby("template_group_id")["email_id"].transform("size")

    duplicate_columns = [
        "email_id",
        "raw_text_hash",
        "sanitised_text_hash",
        "duplicate_group_id",
        "duplicate_group_size",
        "duplicate_label_consistent",
    ]
    template_columns = [
        "email_id",
        "template_group_id",
        "template_signature_hash",
        "template_group_size",
    ]
    duplicate_output = work[duplicate_columns].sort_values("email_id").reset_index(drop=True)
    template_output = work[template_columns].sort_values("email_id").reset_index(drop=True)
    template_output["template_grouping_version"] = "1.0"
    pairs_output = pd.DataFrame(exact_pairs + near_pairs)
    if pairs_output.empty:
        pairs_output = pd.DataFrame(
            columns=["email_id_a", "email_id_b", "cosine_similarity", "evidence_type"]
        )
    else:
        pairs_output = pairs_output.sort_values(
            ["email_id_a", "email_id_b", "evidence_type"]
        ).reset_index(drop=True)

    raw_group_sizes = work.groupby("raw_text_hash").size()
    sanitised_group_sizes = work.groupby("sanitised_text_hash").size()
    mixed_duplicate_groups = int(
        work.loc[work["duplicate_group_size"].gt(1)]
        .groupby("duplicate_group_id")["label_text"]
        .nunique()
        .gt(1)
        .sum()
    )
    summary = {
        "raw_exact_duplicate_groups": int(raw_group_sizes.gt(1).sum()),
        "raw_exact_duplicate_records": int(raw_group_sizes[raw_group_sizes.gt(1)].sum()),
        "sanitised_exact_duplicate_groups": int(sanitised_group_sizes.gt(1).sum()),
        "sanitised_exact_duplicate_records": int(
            sanitised_group_sizes[sanitised_group_sizes.gt(1)].sum()
        ),
        "mixed_label_sanitised_duplicate_groups": mixed_duplicate_groups,
        "template_group_count": int(work["template_group_id"].nunique()),
        "near_similarity_pair_count": len(near_pairs),
        "policy": "keep_all_grouped",
    }
    return GroupingResult(
        frame=work,
        duplicate_groups=duplicate_output,
        template_groups=template_output,
        similarity_pairs=pairs_output,
        summary=summary,
    )

