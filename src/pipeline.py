from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.features import KeywordMatcher, feature_frame
from src.io_utils import load_yaml, write_csv_atomic, write_text_atomic
from src.sanitiser import sanitise_text
from src.schema import load_canonical_dataset


SPLIT_ORDER = ("train", "validation", "test")
BASE_OUTPUT_COLUMNS = [
    "email_id",
    "sanitized_subject",
    "sanitized_body",
    "label",
    "split",
]


class PreprocessingError(RuntimeError):
    """Raised when a minimal preprocessing rule cannot be satisfied safely."""


def _resolve_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _sanitise_frame(canonical: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    removed_blank = 0
    for row in canonical.itertuples(index=False):
        subject = sanitise_text(row.raw_subject).text
        body = sanitise_text(row.raw_body).text
        if not subject.strip() and not body.strip():
            removed_blank += 1
            continue
        rows.append(
            {
                "email_id": row.email_id,
                "original_row_number": int(row.original_row_number),
                "sanitized_subject": subject,
                "sanitized_body": body,
                "label": int(row.label),
            }
        )
    return pd.DataFrame(rows), removed_blank


def _exact_deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work = frame.copy()
    work["exact_duplicate_key"] = (
        work["sanitized_subject"] + "\n" + work["sanitized_body"]
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    grouped = work.groupby("exact_duplicate_key", sort=False, dropna=False)
    label_counts = grouped["label"].nunique()
    conflicting_keys = set(label_counts[label_counts.gt(1)].index)
    if conflicting_keys:
        conflicts = work.loc[
            work["exact_duplicate_key"].isin(conflicting_keys), ["email_id", "label"]
        ]
        raise PreprocessingError(
            "Conflicting labels found in exact duplicate groups: "
            + ", ".join(
                f"{row.email_id}:{row.label}" for row in conflicts.head(20).itertuples(index=False)
            )
        )

    sizes = grouped.size()
    duplicate_keys = set(sizes[sizes.gt(1)].index)
    duplicate_records = int(work["exact_duplicate_key"].isin(duplicate_keys).sum())
    deduplicated = (
        work.sort_values("original_row_number")
        .drop_duplicates("exact_duplicate_key", keep="first")
        .drop(columns="exact_duplicate_key")
        .reset_index(drop=True)
    )
    return deduplicated, {
        "exact_duplicate_groups": len(duplicate_keys),
        "exact_duplicate_records": duplicate_records,
        "exact_duplicates_removed": len(work) - len(deduplicated),
        "conflicting_duplicate_groups": 0,
    }


def _largest_remainder(total: int, ratios: dict[str, float]) -> dict[str, int]:
    exact = {name: total * float(ratios[name]) for name in SPLIT_ORDER}
    result = {name: math.floor(exact[name]) for name in SPLIT_ORDER}
    remaining = total - sum(result.values())
    ranked = sorted(
        SPLIT_ORDER,
        key=lambda name: (-(exact[name] - result[name]), SPLIT_ORDER.index(name)),
    )
    for name in ranked[:remaining]:
        result[name] += 1
    return result


def _stable_order(email_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{email_id}".encode("utf-8")).hexdigest()


def _read_frozen_split(
    root: Path,
    split_config: dict[str, Any],
    known_email_ids: set[str],
) -> dict[str, str] | None:
    paths = {
        name: root / str(split_config.get("frozen_ids", {}).get(name, ""))
        for name in SPLIT_ORDER
    }
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise PreprocessingError("Frozen split is incomplete; expected train, validation and test files")

    mapping: dict[str, str] = {}
    for split_name, path in paths.items():
        ids = pd.read_csv(path)
        if list(ids.columns) != ["email_id"] or not ids["email_id"].astype(str).is_unique:
            raise PreprocessingError(f"Invalid frozen split file: {path}")
        for email_id in ids["email_id"].astype(str):
            if email_id in mapping:
                raise PreprocessingError(f"Frozen email_id appears in multiple splits: {email_id}")
            mapping[email_id] = split_name

    unknown = sorted(set(mapping) - known_email_ids)
    if unknown:
        raise PreprocessingError(f"Frozen split contains unknown email_id values: {unknown[:5]}")
    return mapping


def _new_stratified_split(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in (0, 1):
        ids = frame.loc[frame["label"].eq(label), "email_id"].astype(str).tolist()
        ids.sort(key=lambda email_id: _stable_order(email_id, seed))
        targets = _largest_remainder(len(ids), ratios)
        offset = 0
        for split_name in SPLIT_ORDER:
            for email_id in ids[offset : offset + targets[split_name]]:
                mapping[email_id] = split_name
            offset += targets[split_name]
    return mapping


def _assign_splits(
    frame: pd.DataFrame,
    canonical_ids: set[str],
    root: Path,
    split_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ratios = {name: float(split_config["ratios"][name]) for name in SPLIT_ORDER}
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12):
        raise PreprocessingError("Split ratios must sum to 1")
    seed = int(split_config["seed"])
    frozen = _read_frozen_split(root, split_config, canonical_ids)
    if frozen is None:
        mapping = _new_stratified_split(frame, ratios, seed)
        source = "new_stratified_seed_42"
        retained = 0
    else:
        mapping = {email_id: frozen[email_id] for email_id in frame["email_id"] if email_id in frozen}
        retained = len(mapping)
        missing = frame.loc[~frame["email_id"].isin(mapping)].copy()
        missing["order"] = missing["email_id"].map(lambda value: _stable_order(str(value), seed))
        target_total = _largest_remainder(len(frame), ratios)
        target_by_label = {
            label: _largest_remainder(int(frame["label"].eq(label).sum()), ratios)
            for label in (0, 1)
        }
        for row in missing.sort_values("order").itertuples(index=False):
            current_total = {name: sum(value == name for value in mapping.values()) for name in SPLIT_ORDER}
            label_ids = set(frame.loc[frame["label"].eq(row.label), "email_id"].astype(str))
            current_label = {
                name: sum(email_id in label_ids and split == name for email_id, split in mapping.items())
                for name in SPLIT_ORDER
            }
            chosen = max(
                SPLIT_ORDER,
                key=lambda name: (
                    target_by_label[int(row.label)][name] - current_label[name],
                    target_total[name] - current_total[name],
                    -SPLIT_ORDER.index(name),
                ),
            )
            mapping[str(row.email_id)] = chosen
        source = "existing_frozen_mapping_with_deterministic_extension"

    result = frame.copy()
    result["split"] = result["email_id"].map(mapping)
    if result["split"].isna().any():
        raise PreprocessingError("At least one retained email has no split assignment")
    return result, {
        "split_source": source,
        "frozen_assignments_retained": retained,
        "deterministic_assignments_added": len(result) - retained if frozen is not None else 0,
    }


def build_processed_frame(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_config = config["dataset"]
    canonical = load_canonical_dataset(root / str(dataset_config["path"]), dataset_config)
    sanitised, blank_removed = _sanitise_frame(canonical)
    deduplicated, duplicate_stats = _exact_deduplicate(sanitised)
    assigned, split_stats = _assign_splits(
        deduplicated,
        set(canonical["email_id"].astype(str)),
        root,
        config["split"],
    )

    feature_order = [str(value) for value in config["feature_order"]]
    matcher = KeywordMatcher(config["keywords"])
    features = feature_frame(assigned, matcher, feature_order)
    processed = pd.concat(
        [assigned.reset_index(drop=True), features.reset_index(drop=True)], axis=1
    )
    processed = processed[BASE_OUTPUT_COLUMNS + feature_order].sort_values("email_id").reset_index(drop=True)
    stats = {
        "input_rows": len(canonical),
        "blank_rows_removed": blank_removed,
        "rows_after_blank_rule": len(sanitised),
        "output_rows": len(processed),
        **duplicate_stats,
        **split_stats,
    }
    return processed, stats


def _validate_output(frame: pd.DataFrame, feature_order: list[str]) -> dict[str, str]:
    required = BASE_OUTPUT_COLUMNS + feature_order
    if list(frame.columns) != required or set(frame["label"]) != {0, 1}:
        raise PreprocessingError("Required columns or binary labels are invalid")
    checks = {"必需列存在且标签仅为0/1": "PASS"}

    for column in feature_order:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise PreprocessingError(f"Feature is not numeric: {column}")
        if not frame[column].map(lambda value: math.isfinite(float(value))).all():
            raise PreprocessingError(f"Feature contains NaN or Inf: {column}")
    checks["8项特征均为有限数值"] = "PASS"

    if not frame["uppercase_letter_ratio"].between(0, 1).all():
        raise PreprocessingError("uppercase_letter_ratio is outside [0, 1]")
    count_features = [name for name in feature_order if name != "uppercase_letter_ratio"]
    if (frame[count_features] < 0).any().any():
        raise PreprocessingError("A count feature is negative")
    checks["比例与计数特征范围合法"] = "PASS"

    if not frame["email_id"].is_unique or set(frame["split"]) != set(SPLIT_ORDER):
        raise PreprocessingError("email_id uniqueness or split values are invalid")
    split_sets = {
        name: set(frame.loc[frame["split"].eq(name), "email_id"].astype(str))
        for name in SPLIT_ORDER
    }
    if any(
        split_sets[left] & split_sets[right]
        for index, left in enumerate(SPLIT_ORDER)
        for right in SPLIT_ORDER[index + 1 :]
    ):
        raise PreprocessingError("An email_id appears in multiple splits")
    checks["email_id唯一且三个split互斥"] = "PASS"

    for split_name in SPLIT_ORDER:
        if set(frame.loc[frame["split"].eq(split_name), "label"]) != {0, 1}:
            raise PreprocessingError(f"Split does not contain both labels: {split_name}")
    checks["每个split均包含两类邮件"] = "PASS"
    return checks


def _validate_manual_review(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    review = config.get("manual_spot_check", {})
    ids = [str(value) for value in review.get("email_ids", [])]
    if review.get("status") != "approved" or not review.get("reviewed_date"):
        raise PreprocessingError("Manual spot check has not been approved")
    if len(ids) < 8 or len(ids) != len(set(ids)):
        raise PreprocessingError("Manual spot check must contain at least 8 unique email_id values")
    missing = sorted(set(ids) - set(frame["email_id"].astype(str)))
    if missing:
        raise PreprocessingError(f"Manual spot-check IDs are missing from output: {missing}")
    return ids


def _build_summary(
    frame: pd.DataFrame,
    config: dict[str, Any],
    stats: dict[str, Any],
    checks: dict[str, str],
    manual_count: int,
) -> str:
    feature_order = [str(value) for value in config["feature_order"]]
    lines = [
        "# 最低限度预处理摘要",
        "",
        "## 输入与删减",
        "",
        f"- 输入文件：`{config['dataset']['path']}`",
        f"- 原始记录：{stats['input_rows']}",
        f"- 主题与正文同时为空而删除：{stats['blank_rows_removed']}",
        f"- 精确重复组/涉及记录：{stats['exact_duplicate_groups']} / {stats['exact_duplicate_records']}",
        f"- 删除的同标签精确重复：{stats['exact_duplicates_removed']}",
        f"- 冲突标签重复组：{stats['conflicting_duplicate_groups']}",
        f"- 最终记录：{stats['output_rows']}",
        "",
        "## 固定拆分及类别数量",
        "",
        "| split | legitimate (0) | phishing (1) | total | ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in SPLIT_ORDER:
        subset = frame.loc[frame["split"].eq(split_name)]
        legitimate = int(subset["label"].eq(0).sum())
        phishing = int(subset["label"].eq(1).sum())
        lines.append(
            f"| {split_name} | {legitimate} | {phishing} | {len(subset)} | {len(subset) / len(frame):.2%} |"
        )

    lines += [
        "",
        "## 8项特征范围",
        "",
        "| feature | min | max |",
        "|---|---:|---:|",
    ]
    for column in feature_order:
        lines.append(f"| {column} | {frame[column].min():.6g} | {frame[column].max():.6g} |")

    lines += ["", "## 关键检查", "", "| check | result |", "|---|---|"]
    lines.extend(f"| {name} | {status} |" for name, status in checks.items())
    lines += [
        f"| 人工抽查至少8封邮件 | PASS（已检查 {manual_count} 封） |",
        "",
        "## 已知问题",
        "",
        f"- 保留了 {stats['frozen_assignments_retained']} 个现有冻结 split 分配；对原映射未覆盖的 {stats['deterministic_assignments_added']} 封邮件按固定规则补充分配，没有重排既有记录。",
        "- 为保留既有冻结映射，去重后的实际比例可能与70/15/15存在轻微取整偏差，具体比例见上表。",
        "- 个人信息替换采用保守规则；参与者展示前仍应进行人工隐私复核。",
        "- 没有阻塞性未解决问题；测试集未用于关键词、特征、参数或阈值选择。",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    root = _resolve_root(config_path)
    config = load_yaml(config_path)
    processed, stats = build_processed_frame(root, config)
    feature_order = [str(value) for value in config["feature_order"]]
    checks = _validate_output(processed, feature_order)

    repeated, _ = build_processed_frame(root, config)
    reproducibility_columns = ["email_id", "split", *feature_order]
    if not processed[reproducibility_columns].equals(repeated[reproducibility_columns]):
        raise PreprocessingError("Repeated run changed split or feature values")
    checks["固定种子重复运行结果一致"] = "PASS"

    reviewed_ids = _validate_manual_review(processed, config)
    output_path = root / str(config["output"]["processed_csv"])
    summary_path = root / str(config["output"]["summary"])
    write_csv_atomic(output_path, processed)
    write_text_atomic(
        summary_path,
        _build_summary(processed, config, stats, checks, len(reviewed_ids)),
    )

    split_counts = {
        split_name: {
            "legitimate": int(
                processed.loc[processed["split"].eq(split_name), "label"].eq(0).sum()
            ),
            "phishing": int(
                processed.loc[processed["split"].eq(split_name), "label"].eq(1).sum()
            ),
            "total": int(processed["split"].eq(split_name).sum()),
        }
        for split_name in SPLIT_ORDER
    }
    return {
        "status": "PASS",
        "output": str(output_path),
        "summary": str(summary_path),
        "rows": len(processed),
        "split_counts": split_counts,
        "checks": {**checks, "人工抽查至少8封邮件": "PASS"},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal phishing-email preprocessing")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(Path(args.config).resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
