from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import KeywordMatcher, feature_frame
from src.io_utils import load_yaml


DEFAULT_KS_THRESHOLD = 0.20
DEFAULT_NEAR_CONSTANT_RATIO = 0.995
SUMMARY_COLUMNS = [
    "min",
    "max",
    "mean",
    "median",
    "std",
    "nonzero_count",
    "nonzero_ratio",
    "unique_count",
    "mode_ratio",
]


class FeatureDiagnosticError(RuntimeError):
    """Raised when the processed feature table cannot be diagnosed safely."""


def _distribution_summary(
    frame: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for feature in feature_order:
        values = frame[feature]
        mode_ratio = float(values.value_counts(normalize=True, dropna=False).iloc[0])
        rows.append(
            {
                "feature": feature,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)),
                "nonzero_count": int(values.ne(0).sum()),
                "nonzero_ratio": float(values.ne(0).mean()),
                "unique_count": int(values.nunique(dropna=False)),
                "mode_ratio": mode_ratio,
            }
        )
    return pd.DataFrame(rows).set_index("feature")[SUMMARY_COLUMNS]


def _grouped_distribution(
    frame: pd.DataFrame,
    group_column: str,
    feature_order: list[str],
) -> pd.DataFrame:
    groups: list[pd.DataFrame] = []
    for group_value in sorted(frame[group_column].unique(), key=str):
        subset = frame.loc[frame[group_column].eq(group_value), feature_order]
        summary = _distribution_summary(subset, feature_order).reset_index()
        summary.insert(0, group_column, group_value)
        groups.append(summary)
    return pd.concat(groups, ignore_index=True).set_index([group_column, "feature"])


def _ks_distance(left: pd.Series, right: pd.Series) -> float:
    left_values = np.sort(left.to_numpy(dtype=float))
    right_values = np.sort(right.to_numpy(dtype=float))
    if not len(left_values) or not len(right_values):
        raise FeatureDiagnosticError("Cannot compare empty split distributions")
    support = np.sort(np.unique(np.concatenate([left_values, right_values])))
    left_cdf = np.searchsorted(left_values, support, side="right") / len(left_values)
    right_cdf = np.searchsorted(right_values, support, side="right") / len(right_values)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _split_distances(frame: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    rows: list[dict[str, float | str]] = []
    for feature in feature_order:
        for left, right in pairs:
            rows.append(
                {
                    "feature": feature,
                    "split_pair": f"{left} vs {right}",
                    "ks_distance": _ks_distance(
                        frame.loc[frame["split"].eq(left), feature],
                        frame.loc[frame["split"].eq(right), feature],
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["feature", "split_pair"])


def analyse_feature_validity(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    ks_threshold: float = DEFAULT_KS_THRESHOLD,
    near_constant_ratio: float = DEFAULT_NEAR_CONSTANT_RATIO,
) -> dict[str, Any]:
    feature_order = [str(value) for value in config.get("feature_order", [])]
    if len(feature_order) != 8 or len(set(feature_order)) != 8:
        raise FeatureDiagnosticError("Configuration must define exactly eight unique features")

    required = {
        "email_id",
        "sanitized_subject",
        "sanitized_body",
        "label",
        "split",
        *feature_order,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FeatureDiagnosticError(f"Processed data is missing columns: {missing}")

    numeric = frame[feature_order].apply(pd.to_numeric, errors="raise")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    nonfinite_count = int((~finite_mask).sum())

    matcher = KeywordMatcher(config["keywords"])
    recomputed = feature_frame(frame, matcher, feature_order)
    mismatch_counts: dict[str, int] = {}
    for feature in feature_order:
        stored = numeric[feature].to_numpy(dtype=float)
        expected = recomputed[feature].to_numpy(dtype=float)
        if feature == "uppercase_letter_ratio":
            matches = np.isclose(stored, expected, rtol=0.0, atol=1e-12)
        else:
            matches = stored == expected
        mismatch_counts[feature] = int((~matches).sum())

    diagnostic_frame = frame.copy()
    diagnostic_frame[feature_order] = numeric
    overall = _distribution_summary(diagnostic_frame, feature_order)
    by_label = _grouped_distribution(diagnostic_frame, "label", feature_order)
    by_split = _grouped_distribution(diagnostic_frame, "split", feature_order)
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    split_distances = _split_distances(diagnostic_frame, feature_order)

    constant_features = overall.index[overall["unique_count"].le(1)].tolist()
    nearly_constant_features = overall.index[
        overall["mode_ratio"].ge(near_constant_ratio)
    ].tolist()
    split_anomalies = split_distances.loc[
        split_distances["ks_distance"].ge(ks_threshold)
    ].reset_index().to_dict("records")
    total_mismatches = sum(mismatch_counts.values())
    passed = (
        not constant_features
        and not split_anomalies
        and not nonfinite_count
        and not total_mismatches
    )

    return {
        "passed": passed,
        "feature_order": feature_order,
        "overall": overall,
        "by_label": by_label,
        "by_split": by_split,
        "pearson": pearson,
        "spearman": spearman,
        "split_distances": split_distances,
        "ks_threshold": ks_threshold,
        "near_constant_ratio": near_constant_ratio,
        "constant_features": constant_features,
        "nearly_constant_features": nearly_constant_features,
        "split_anomalies": split_anomalies,
        "mismatch_counts": mismatch_counts,
        "nonfinite_count": nonfinite_count,
        "url_feature_mismatches": mismatch_counts["url_count"]
        + mismatch_counts["ip_address_url_count"],
        "keyword_feature_mismatches": sum(
            mismatch_counts[name]
            for name in (
                "urgency_word_count",
                "credential_word_count",
                "action_word_count",
                "money_related_word_count",
            )
        ),
    }


def print_feature_diagnostics(result: dict[str, Any]) -> None:
    float_format = lambda value: f"{value:.6f}"
    print("Frozen feature order:")
    for index, feature in enumerate(result["feature_order"], start=1):
        print(f"  {index}. {feature}")
    print("\nOverall distribution:")
    print(result["overall"].to_string(float_format=float_format))
    print("\nDistribution by label:")
    print(result["by_label"].to_string(float_format=float_format))
    print("\nDistribution by split:")
    print(result["by_split"].to_string(float_format=float_format))
    print("\nPearson correlation:")
    print(result["pearson"].to_string(float_format=float_format))
    print("\nSpearman correlation:")
    print(result["spearman"].to_string(float_format=float_format))
    print(f"\nCross-split KS distances (alert threshold={result['ks_threshold']:.2f}):")
    print(result["split_distances"].to_string(float_format=float_format))
    print("\nRecomputation mismatch counts:")
    for feature, count in result["mismatch_counts"].items():
        print(f"  {feature}: {count}")
    if result["nearly_constant_features"]:
        print(
            "Sparse/nearly constant warning (not an automatic failure): "
            + ", ".join(result["nearly_constant_features"])
        )
    print(f"FEATURE VALIDITY: {'PASS' if result['passed'] else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose the frozen eight preprocessing features")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/processed_emails.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    parser.add_argument("--ks-threshold", type=float, default=DEFAULT_KS_THRESHOLD)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_yaml(args.config.resolve())
        frame = pd.read_csv(args.input.resolve(), keep_default_na=False)
        result = analyse_feature_validity(frame, config, ks_threshold=args.ks_threshold)
        print_feature_diagnostics(result)
    except (FeatureDiagnosticError, KeyError, TypeError, ValueError) as error:
        print(f"Feature diagnostics failed: {error}", file=sys.stderr)
        return 1
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
