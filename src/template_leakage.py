from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


RANDOM_SEED = 42
SIMILARITY_THRESHOLDS = (0.90, 0.95)
SPLIT_PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
DEFAULT_BLOCK_SIZE = 128
DEFAULT_SEVERE_EVALUATION_COVERAGE = 0.05


class TemplateLeakageError(RuntimeError):
    """Raised when template leakage diagnostics cannot be performed safely."""


@dataclass(frozen=True)
class SimilarEmailPair:
    left_email_id: str
    left_split: str
    left_label: int
    right_email_id: str
    right_split: str
    right_label: int
    similarity: float


def _combined_text(frame: pd.DataFrame) -> pd.Series:
    return frame["sanitized_subject"].astype(str) + "\n" + frame["sanitized_body"].astype(str)


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        min_df=1,
        norm="l2",
        dtype=np.float64,
    )


def analyse_template_leakage(
    frame: pd.DataFrame,
    *,
    thresholds: tuple[float, float] = SIMILARITY_THRESHOLDS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    random_seed: int = RANDOM_SEED,
    severe_evaluation_coverage: float = DEFAULT_SEVERE_EVALUATION_COVERAGE,
) -> dict[str, Any]:
    required = {"email_id", "sanitized_subject", "sanitized_body", "label", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise TemplateLeakageError(f"Processed data is missing columns: {missing}")
    if tuple(sorted(thresholds)) != thresholds or thresholds != SIMILARITY_THRESHOLDS:
        raise TemplateLeakageError("Similarity thresholds must remain fixed at 0.90 and 0.95")
    if block_size <= 0:
        raise TemplateLeakageError("block_size must be positive")
    if frame["email_id"].duplicated().any():
        raise TemplateLeakageError("email_id must be unique")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise TemplateLeakageError("Expected train, validation, and test splits")

    # The method is deterministic; the seed is fixed and recorded for reproducibility.
    np.random.default_rng(random_seed)
    texts = _combined_text(frame)
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(texts)
    split_values = frame["split"].to_numpy()
    minimum_threshold = thresholds[0]
    pairs: list[SimilarEmailPair] = []

    for left_split, right_split in SPLIT_PAIRS:
        left_indices = np.flatnonzero(split_values == left_split)
        right_indices = np.flatnonzero(split_values == right_split)
        for start in range(0, len(left_indices), block_size):
            left_block = left_indices[start : start + block_size]
            similarities = cosine_similarity(matrix[left_block], matrix[right_indices])
            block_rows, right_columns = np.where(similarities >= minimum_threshold)
            for block_row, right_column in zip(block_rows, right_columns, strict=True):
                left_index = int(left_block[block_row])
                right_index = int(right_indices[right_column])
                left = frame.iloc[left_index]
                right = frame.iloc[right_index]
                pairs.append(
                    SimilarEmailPair(
                        left_email_id=str(left["email_id"]),
                        left_split=str(left["split"]),
                        left_label=int(left["label"]),
                        right_email_id=str(right["email_id"]),
                        right_split=str(right["split"]),
                        right_label=int(right["label"]),
                        similarity=float(similarities[block_row, right_column]),
                    )
                )

    pairs.sort(key=lambda pair: (-pair.similarity, pair.left_email_id, pair.right_email_id))
    threshold_summaries: dict[str, dict[str, int | float]] = {}
    evaluation_ids = set(frame.loc[frame["split"].ne("train"), "email_id"].astype(str))
    high_threshold = thresholds[1]
    high_pairs = [pair for pair in pairs if pair.similarity >= high_threshold]
    high_evaluation_ids = {
        email_id
        for pair in high_pairs
        for email_id, split in (
            (pair.left_email_id, pair.left_split),
            (pair.right_email_id, pair.right_split),
        )
        if split != "train"
    }

    for threshold in thresholds:
        selected = [pair for pair in pairs if pair.similarity >= threshold]
        related_evaluation_ids = {
            email_id
            for pair in selected
            for email_id, split in (
                (pair.left_email_id, pair.left_split),
                (pair.right_email_id, pair.right_split),
            )
            if split != "train"
        }
        threshold_summaries[f"{threshold:.2f}"] = {
            "pair_count": len(selected),
            "same_label_pairs": sum(pair.left_label == pair.right_label for pair in selected),
            "opposite_label_pairs": sum(pair.left_label != pair.right_label for pair in selected),
            "evaluation_email_count": len(related_evaluation_ids),
            "evaluation_coverage": len(related_evaluation_ids) / len(evaluation_ids),
        }

    text_lookup = dict(zip(frame["email_id"].astype(str), texts, strict=True))
    exact_cross_split_pairs = [
        pair
        for pair in pairs
        if text_lookup[pair.left_email_id] == text_lookup[pair.right_email_id]
    ]
    opposite_label_high_pairs = [
        pair for pair in high_pairs if pair.left_label != pair.right_label
    ]
    high_evaluation_coverage = len(high_evaluation_ids) / len(evaluation_ids)
    severe = bool(
        exact_cross_split_pairs
        or opposite_label_high_pairs
        or high_evaluation_coverage >= severe_evaluation_coverage
    )

    return {
        "passed": not severe,
        "severe_leakage": severe,
        "may_inflate_performance": any(
            pair.left_label == pair.right_label for pair in pairs
        ),
        "random_seed": random_seed,
        "method": "word (1,2)-gram TF-IDF cosine similarity",
        "vectorizer_parameters": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "lowercase": True,
            "strip_accents": "unicode",
            "sublinear_tf": True,
            "min_df": 1,
            "norm": "l2",
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "threshold_summaries": threshold_summaries,
        "pairs": pairs,
        "exact_cross_split_pair_count": len(exact_cross_split_pairs),
        "opposite_label_high_pair_count": len(opposite_label_high_pairs),
        "high_evaluation_coverage": high_evaluation_coverage,
        "severe_evaluation_coverage": severe_evaluation_coverage,
    }


def print_template_leakage(result: dict[str, Any]) -> None:
    print(f"Method: {result['method']}")
    print(f"Random seed: {result['random_seed']}")
    print(f"Vocabulary size: {result['vocabulary_size']}")
    print("\nThreshold summary:")
    summary_rows = []
    for threshold, summary in result["threshold_summaries"].items():
        summary_rows.append({"threshold": threshold, **summary})
    summary_frame = pd.DataFrame(summary_rows).set_index("threshold")
    print(summary_frame.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nCross-split pairs with similarity >= 0.90:")
    if result["pairs"]:
        pair_frame = pd.DataFrame(asdict(pair) for pair in result["pairs"])
        print(pair_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        print("  none")

    print(f"\nExact sanitized-text cross-split pairs: {result['exact_cross_split_pair_count']}")
    print(f"Opposite-label pairs at >= 0.95: {result['opposite_label_high_pair_count']}")
    print(
        "Evaluation-email coverage at >= 0.95: "
        f"{result['high_evaluation_coverage']:.4%} "
        f"(severe threshold={result['severe_evaluation_coverage']:.2%})"
    )
    print(f"TEMPLATE LEAKAGE: {'PASS' if result['passed'] else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose cross-split near duplicates")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed_emails.csv"),
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = pd.read_csv(args.input.resolve(), keep_default_na=False)
        result = analyse_template_leakage(frame, block_size=args.block_size)
        print_template_leakage(result)
    except (TemplateLeakageError, KeyError, TypeError, ValueError) as error:
        print(f"Template leakage diagnostics failed: {error}", file=sys.stderr)
        return 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
