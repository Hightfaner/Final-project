from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from src.audit import audit_markdown, build_dataset_audit
from src.features import FEATURE_ORDER, KeywordMatcher, feature_frame
from src.io_utils import (
    JsonlLogger,
    canonical_json_hash,
    ensure_directories,
    file_hash,
    load_json,
    load_yaml,
    utc_now,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from src.sanitiser import meaningful_body, sanitise_text
from src.schema import DatasetSnapshot, load_canonical_dataset
from src.splitter import SPLIT_ORDER, build_fixed_split
from src.template_grouping import GroupingResult, build_groups
from src.validation import (
    FINAL_MATRIX_COLUMNS,
    ValidationError,
    validate_feature_contract,
    validate_feature_values,
    validate_output_bundle,
    validate_sanitised_content,
)


def _resolve_root(config_path: Path) -> Path:
    return config_path.resolve().parent.parent


def _sanitise_frame(canonical: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in canonical.itertuples(index=False):
        subject = sanitise_text(row.raw_subject)
        body = sanitise_text(row.raw_body)
        if sanitise_text(subject.text).text != subject.text:
            raise ValidationError(f"Subject sanitisation is not idempotent for {row.email_id}")
        if sanitise_text(body.text).text != body.text:
            raise ValidationError(f"Body sanitisation is not idempotent for {row.email_id}")
        eligible = meaningful_body(body.text)
        reason = "" if eligible else "body_missing_or_empty_after_sanitisation"
        record = {
            "email_id": row.email_id,
            "original_row_number": int(row.original_row_number),
            "label_original": int(row.label_original),
            "label_text": row.label_text,
            "sanitised_subject": subject.text,
            "sanitised_body": body.text,
            "sanitisation_version": version,
            "url_replacement_count": subject.url_replacement_count + body.url_replacement_count,
            "pii_replacement_count": subject.pii_replacement_count + body.pii_replacement_count,
            "removed_active_content_count": (
                subject.removed_active_content_count + body.removed_active_content_count
            ),
            "processing_status": "eligible" if eligible else "excluded",
            "exclusion_reason": reason,
        }
        rows.append(record)
        if not eligible:
            exclusions.append(
                {
                    "email_id": row.email_id,
                    "original_row_number": int(row.original_row_number),
                    "label_original": int(row.label_original),
                    "stage": "sanitisation",
                    "reason": reason,
                    "rule_version": version,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(
        exclusions,
        columns=[
            "email_id",
            "original_row_number",
            "label_original",
            "stage",
            "reason",
            "rule_version",
        ],
    )


def _feature_report(matrix: pd.DataFrame, keyword_config: dict[str, Any]) -> str:
    lines = [
        "# Final Eight-Feature Report",
        "",
        f"- Feature contract: version 1.0, status frozen",
        f"- Keyword version: {keyword_config['keywords_version']}",
        f"- Keyword status: {keyword_config['keywords_status']}",
        f"- Frozen date: {keyword_config['frozen_date']}",
        "- Text source: `sanitised_subject` + newline + `sanitised_body`",
        "",
        "## Feature statistics",
        "",
        "| Feature | Min | Max | Mean | Median | Zero ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for column in FEATURE_ORDER:
        values = matrix[column]
        lines.append(
            f"| {column} | {values.min():.6g} | {values.max():.6g} | "
            f"{values.mean():.6g} | {values.median():.6g} | {values.eq(0).mean():.6f} |"
        )
    lines += [
        "",
        "## Keyword-feature non-zero email counts by ground-truth label",
        "",
        "These descriptive counts are reported only after the researcher-approved freeze and were not used to select or modify any keyword.",
        "",
        "| Feature | phishing | legitimate |",
        "|---|---:|---:|",
    ]
    for column in FEATURE_ORDER[2:6]:
        counts = matrix.assign(nonzero=matrix[column].gt(0)).groupby("label_text")["nonzero"].sum()
        lines.append(
            f"| {column} | {int(counts.get('phishing', 0))} | {int(counts.get('legitimate', 0))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _keyword_freeze_report(keyword_config: dict[str, Any]) -> str:
    config_hash = canonical_json_hash(keyword_config)
    lines = [
        "# Keyword Freeze Report",
        "",
        f"- Version: {keyword_config['keywords_version']}",
        f"- Status: {keyword_config['keywords_status']}",
        f"- Frozen date: {keyword_config['frozen_date']}",
        f"- Canonical configuration SHA-256: `{config_hash}`",
        "",
        "The lists were supplied as a researcher-approved semantic operationalisation before model evaluation. No validation/test label-conditional frequency, hit-rate, discrimination statistic, or model result was used.",
        "",
        "## Matching contract",
        "",
        "- Case-insensitive temporary matching copy; persisted sanitised text is unchanged.",
        "- Whole-word/whole-phrase, phrase-first, non-overlapping occurrence counts.",
        "- Fixed variant normalisation only; no stemming, lemmatisation, fuzzy matching, or expansion.",
        "- Input fields are only `sanitised_subject` and `sanitised_body`.",
        "",
    ]
    for category in ("urgency", "credential", "action", "money_related"):
        lines += [f"## {category}", "", ", ".join(keyword_config[category]), ""]
    return "\n".join(lines)


def _review_report(grouping: GroupingResult) -> str:
    mixed = grouping.frame.loc[
        grouping.frame["duplicate_group_size"].gt(1)
        & ~grouping.frame["duplicate_label_consistent"],
        ["email_id", "label_text", "duplicate_group_id"],
    ].sort_values(["duplicate_group_id", "email_id"])
    lines = [
        "# Researcher Review Required",
        "",
        "Sanitised duplicate groups are retained under `keep_all_grouped`; no record is deleted, merged, or relabelled.",
        "",
    ]
    if mixed.empty:
        lines.append("No sanitised exact-duplicate group contains mixed ground-truth labels.")
    else:
        lines += [
            "The following sanitised exact-duplicate groups contain mixed ground-truth labels and require researcher review. They remain grouped in one split.",
            "",
            "| duplicate_group_id | email_id | label_text |",
            "|---|---|---|",
        ]
        for row in mixed.itertuples(index=False):
            lines.append(f"| {row.duplicate_group_id} | {row.email_id} | {row.label_text} |")
    lines.append("")
    return "\n".join(lines)


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted-initial-implementation"


def _write_checksums(root: Path, reports_root: Path) -> dict[str, str]:
    candidates = [
        root / "README.md",
        root / "requirements.txt",
        root / "pyproject.toml",
        root / "run_pipeline.py",
        root / "configs" / "pipeline.yaml",
        root / "configs" / "column_mapping.yaml",
        root / "configs" / "keywords.json",
        root / "configs" / "feature_contract.json",
        root / "data" / "raw" / "Nazario_5.csv",
        *sorted((root / "data" / "interim").glob("*")),
        *sorted((root / "data" / "splits").glob("*")),
        *sorted((root / "data" / "processed").glob("*")),
        *sorted(path for path in reports_root.glob("*.md")),
        *sorted(path for path in reports_root.glob("*.json") if path.name != "output_checksums.json"),
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "tests").rglob("*.py")),
    ]
    checksum_path = reports_root / "output_checksums.json"
    hashes = {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in candidates
        if path.is_file() and path != checksum_path
    }
    write_json_atomic(checksum_path, {"algorithm": "sha256", "sha256": hashes})
    return hashes


def _implementation_report(
    snapshot: DatasetSnapshot,
    grouping: GroupingResult,
    split_summary: dict[str, Any],
    matrix: pd.DataFrame,
    matrix_paths: dict[str, Path],
    raw_hash_after: str,
    test_result: dict[str, Any],
) -> str:
    matrix_details = [
        (name, len(pd.read_csv(path)), len(pd.read_csv(path, nrows=0).columns), file_hash(path))
        for name, path in matrix_paths.items()
    ]
    lines = [
        "# Week 6–7 Final Implementation Report",
        "",
        "## Commands and exit status",
        "",
        f"- `python -m pytest -q`: PASS (exit {test_result['exit_code']}; {test_result['passed']} passed).",
        "- `python run_pipeline.py --config configs/pipeline.yaml --mode final`: PASS (exit 0).",
        "- `python run_pipeline.py --config configs/pipeline.yaml --stage validate-only`: pending the required post-final validation command.",
        "",
        "## Frozen invariants",
        "",
        f"- Raw SHA-256 before: `{snapshot.sha256}`",
        f"- Raw SHA-256 after: `{raw_hash_after}`",
        f"- Eligible/excluded: {len(matrix)} / {snapshot.row_count - len(matrix)}",
        f"- Split counts: {split_summary['actual_counts']}",
        f"- Cross-split template groups: {split_summary['cross_split_template_group_violations']}",
        f"- Raw exact duplicate groups/records: {grouping.summary['raw_exact_duplicate_groups']} / {grouping.summary['raw_exact_duplicate_records']}",
        f"- Sanitised exact duplicate groups/records: {grouping.summary['sanitised_exact_duplicate_groups']} / {grouping.summary['sanitised_exact_duplicate_records']}",
        f"- Mixed-label sanitised duplicate groups: {grouping.summary['mixed_label_sanitised_duplicate_groups']}",
        "- Duplicate policy: `keep_all_grouped`.",
        "- Keyword configuration: version 1.0, status frozen, date 2026-07-30.",
        "- Feature contract: exactly eight columns in the frozen order.",
        "",
        "## Final matrices",
        "",
        "| Matrix | Rows | Columns | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, rows, columns, digest in matrix_details:
        lines.append(f"| {name} | {rows} | {columns} | `{digest}` |")
    lines += [
        "",
        "## Definition of Done",
        "",
        "| Item | Status | Evidence |",
        "|---|---|---|",
        "| Dataset and raw protection | PASS | Read-only raw file; before/after SHA-256 identical. |",
        "| Labels | PASS | Only configured 0=legitimate and 1=phishing mappings accepted. |",
        "| Audit and sanitisation | PASS | Dataset audit, deterministic/idempotent safe text, exclusions recorded. |",
        "| Duplicate/template grouping | PASS | Content-only grouping; keep-all policy; review report generated. |",
        "| Fixed split | PASS | 2144/460/459, disjoint union, zero cross-group violations. |",
        "| Frozen eight features | PASS | Names, order, source, ranges and model interface validated. |",
        "| Frozen keywords | PASS | Version/status/date/lists/rules match the approved specification. |",
        "| Final outputs | PASS | Non-provisional all/train/validation/test matrices generated. |",
        f"| Automated tests | PASS | {test_result['passed']} tests passed with exit {test_result['exit_code']}. |",
        "| Validate-only | PENDING | Recorded after the required post-final command. |",
        "",
        "## Remaining risks",
        "",
        "- Any mixed-label sanitised duplicate group listed in `researcher_review_required.md` remains a researcher decision; records stay grouped and unchanged.",
        "- Near-template grouping uses the documented engineering defaults (seed 42, char 3–5gram TF-IDF, cosine >=0.95) and is not a model-performance choice.",
        "",
    ]
    return "\n".join(lines)


def run_final(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    root = _resolve_root(config_path)
    config = load_yaml(config_path)
    output_root = root / str(config["output_root"])
    reports_root = root / str(config.get("reports_root", "reports"))
    log_path = root / str(config.get("log_path", "logs/pipeline.jsonl"))
    dataset_path = root / str(config["dataset_path"])
    ensure_directories(
        output_root / "interim",
        output_root / "splits",
        output_root / "processed",
        reports_root,
        log_path.parent,
    )
    final_guard = output_root / "processed" / "feature_matrix_all.csv"
    if final_guard.exists() and config.get("output", {}).get("refuse_final_overwrite", True) and not overwrite:
        raise FileExistsError(
            f"Final outputs already exist at {final_guard}; rerun with explicit --overwrite only after verifying the manifest"
        )
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if bool(dataset_path.stat().st_mode & stat.S_IWRITE):
        raise PermissionError(f"Raw dataset must be read-only before final execution: {dataset_path}")

    run_id = str(uuid.uuid4())
    logger = JsonlLogger(log_path, run_id)
    logger.event(
        "pipeline",
        "started",
        mode="final",
        command="run_pipeline.py --config configs/pipeline.yaml --mode final",
    )
    raw_hash_before = file_hash(dataset_path)
    mapping = load_yaml(root / str(config["column_mapping"]))
    feature_contract = load_json(root / str(config["feature_config"]))
    keywords = load_json(root / str(config["keyword_config"]))
    validate_feature_contract(feature_contract)
    matcher = KeywordMatcher(keywords)
    test_result_path = reports_root / "test_results.json"
    if not test_result_path.is_file():
        raise ValidationError("Required pre-final pytest evidence is missing: reports/test_results.json")
    test_result = load_json(test_result_path)
    if test_result.get("status") != "PASS" or int(test_result.get("exit_code", 1)) != 0:
        raise ValidationError("Pre-final pytest evidence is not PASS")

    canonical, snapshot = load_canonical_dataset(
        dataset_path,
        mapping,
        str(config["dataset"]["encoding"]),
        int(config["dataset"]["expected_raw_rows"]),
    )
    audit = build_dataset_audit(canonical, snapshot)
    logger.event("input_audit", "passed", rows=len(canonical), dataset_sha256=snapshot.sha256)

    sanitised, exclusions = _sanitise_frame(canonical, str(config["sanitisation"]["version"]))
    validate_sanitised_content(sanitised)
    eligible_count = int(sanitised["processing_status"].eq("eligible").sum())
    expected_eligible = int(config["dataset"]["expected_eligible_rows"])
    if eligible_count != expected_eligible:
        raise ValidationError(f"Eligible row count {eligible_count} differs from expected {expected_eligible}")
    logger.event("sanitisation", "passed", eligible=eligible_count, excluded=len(exclusions))

    eligible_ids = set(sanitised.loc[sanitised["processing_status"].eq("eligible"), "email_id"])
    grouping_input = canonical.loc[canonical["email_id"].isin(eligible_ids)].copy()
    grouping_input = grouping_input.merge(
        sanitised[["email_id", "sanitised_subject", "sanitised_body"]],
        on="email_id",
        how="left",
        validate="one_to_one",
    )
    grouping = build_groups(grouping_input, config["template_grouping"])
    logger.event("grouping", "passed", **grouping.summary)

    split_result = build_fixed_split(grouping.frame, config["split"])
    logger.event("split", "passed", **split_result.summary)

    eligible = sanitised.loc[sanitised["processing_status"].eq("eligible")].copy()
    eligible = eligible.merge(
        grouping.template_groups[["email_id", "template_group_id"]],
        on="email_id",
        how="left",
        validate="one_to_one",
    ).merge(
        split_result.assignments,
        on="email_id",
        how="left",
        validate="one_to_one",
    )
    features = feature_frame(eligible, matcher)
    matrix = pd.concat([eligible.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    matrix["feature_contract_version"] = str(feature_contract["version"])
    matrix = matrix[FINAL_MATRIX_COLUMNS].sort_values("email_id").reset_index(drop=True)
    validate_feature_values(matrix)
    logger.event("features", "passed", rows=len(matrix), feature_columns=FEATURE_ORDER)

    interim = output_root / "interim"
    splits = output_root / "splits"
    processed = output_root / "processed"
    write_csv_atomic(interim / "nazario_5_sanitised.csv", sanitised.sort_values("email_id"))
    write_csv_atomic(interim / "duplicate_groups.csv", grouping.duplicate_groups)
    write_csv_atomic(interim / "template_groups.csv", grouping.template_groups)
    write_csv_atomic(interim / "template_similarity_pairs.csv", grouping.similarity_pairs)
    write_csv_atomic(interim / "exclusions.csv", exclusions.sort_values("email_id"))

    for split_name in SPLIT_ORDER:
        ids = (
            split_result.assignments.loc[split_result.assignments["split"].eq(split_name), ["email_id"]]
            .sort_values("email_id")
            .reset_index(drop=True)
        )
        write_csv_atomic(splits / f"{split_name}_ids.csv", ids)

    matrix_paths = {"all": processed / "feature_matrix_all.csv"}
    write_csv_atomic(matrix_paths["all"], matrix)
    for split_name in SPLIT_ORDER:
        path = processed / f"feature_matrix_{split_name}.csv"
        subset = matrix.loc[matrix["split"].eq(split_name)].sort_values("email_id").reset_index(drop=True)
        write_csv_atomic(path, subset)
        matrix_paths[split_name] = path

    raw_hash_after = file_hash(dataset_path)
    if raw_hash_after != raw_hash_before:
        raise ValidationError("Raw dataset SHA-256 changed during final processing")
    manifest = {
        "manifest_version": "1.0",
        "created_utc": utc_now(),
        "dataset_path": str(config["dataset_path"]),
        "dataset_md5": snapshot.md5,
        "dataset_sha256": snapshot.sha256,
        "dataset_rows": snapshot.row_count,
        "eligible_rows": len(matrix),
        "excluded_rows": len(exclusions),
        "sanitisation_version": str(config["sanitisation"]["version"]),
        "template_grouping_version": "1.0",
        "feature_contract_version": feature_contract["version"],
        "keywords_version": keywords["keywords_version"],
        "keywords_status": keywords["keywords_status"],
        "keywords_frozen_date": keywords["frozen_date"],
        "seed": int(config["split"]["seed"]),
        "target_ratios": {
            "train": float(config["split"]["train_ratio"]),
            "validation": float(config["split"]["validation_ratio"]),
            "test": float(config["split"]["test_ratio"]),
        },
        "split_summary": split_result.summary,
        "duplicate_summary": grouping.summary,
        "config_hashes": {
            "pipeline.yaml": file_hash(config_path),
            "column_mapping.yaml": file_hash(root / str(config["column_mapping"])),
            "feature_contract.json": file_hash(root / str(config["feature_config"])),
            "keywords.json": file_hash(root / str(config["keyword_config"])),
        },
        "artifact_sha256": {
            path.relative_to(root).as_posix(): file_hash(path)
            for path in [
                interim / "nazario_5_sanitised.csv",
                interim / "duplicate_groups.csv",
                interim / "template_groups.csv",
                interim / "template_similarity_pairs.csv",
                interim / "exclusions.csv",
                splits / "train_ids.csv",
                splits / "validation_ids.csv",
                splits / "test_ids.csv",
                *matrix_paths.values(),
            ]
        },
        "code_revision": _git_revision(root),
    }
    write_json_atomic(splits / "split_manifest.json", manifest)
    write_json_atomic(reports_root / "dataset_audit.json", audit)
    write_text_atomic(reports_root / "dataset_audit.md", audit_markdown(audit))
    write_text_atomic(reports_root / "feature_report.md", _feature_report(matrix, keywords))
    write_text_atomic(reports_root / "keyword_freeze_report.md", _keyword_freeze_report(keywords))
    write_text_atomic(reports_root / "researcher_review_required.md", _review_report(grouping))
    implementation = _implementation_report(
        snapshot,
        grouping,
        split_result.summary,
        matrix,
        matrix_paths,
        raw_hash_after,
        test_result,
    )
    write_text_atomic(reports_root / "implementation_report.md", implementation)
    hashes = _write_checksums(root, reports_root)
    logger.event("pipeline", "passed", mode="final", exit_code=0, output_file_count=len(hashes))
    return {
        "status": "PASS",
        "rows": len(matrix),
        "split_counts": split_result.summary["actual_counts"],
        "dataset_sha256": snapshot.sha256,
        "grouping": grouping.summary,
    }


def run_validate_only(config_path: Path) -> dict[str, Any]:
    root = _resolve_root(config_path)
    config = load_yaml(config_path)
    reports_root = root / str(config.get("reports_root", "reports"))
    log_path = root / str(config.get("log_path", "logs/pipeline.jsonl"))
    logger = JsonlLogger(log_path, str(uuid.uuid4()))
    logger.event(
        "validate_only",
        "started",
        command="run_pipeline.py --config configs/pipeline.yaml --stage validate-only",
    )
    result = validate_output_bundle(root, config)
    report = "\n".join(
        [
            "# Validate-Only Report",
            "",
            "- Status: **PASS**",
            f"- Dataset SHA-256: `{result['dataset_sha256']}`",
            f"- Raw / eligible / excluded rows: {result['raw_rows']} / {result['eligible_rows']} / {result['excluded_rows']}",
            f"- Split counts: {result['split_counts']}",
            f"- Cross-split template groups: {result['cross_split_template_group_violations']}",
            f"- Matrix shapes: {result['matrix_shapes']}",
            f"- Keyword non-zero rows: {result['keyword_nonzero_rows']}",
            "",
        ]
    )
    write_text_atomic(reports_root / "validation_report.md", report)
    implementation_path = reports_root / "implementation_report.md"
    implementation = implementation_path.read_text(encoding="utf-8")
    implementation = implementation.replace(
        "- `python run_pipeline.py --config configs/pipeline.yaml --stage validate-only`: pending the required post-final validation command.",
        "- `python run_pipeline.py --config configs/pipeline.yaml --stage validate-only`: PASS (exit 0).",
    ).replace(
        "| Validate-only | PENDING | Recorded after the required post-final command. |",
        "| Validate-only | PASS | Required post-final bundle validation completed with exit 0. |",
    )
    write_text_atomic(implementation_path, implementation)
    _write_checksums(root, reports_root)
    logger.event(
        "validate_only",
        "passed",
        exit_code=0,
        validation_status=result["status"],
        **{key: value for key, value in result.items() if key != "status"},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Week 6–7 phishing-email data pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--mode", choices=["final"])
    selector.add_argument("--stage", choices=["validate-only"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing final outputs after manifest review",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    try:
        if args.stage == "validate-only":
            result = run_validate_only(config_path)
        else:
            result = run_final(config_path, overwrite=bool(args.overwrite))
    except Exception as exc:
        try:
            failed_config = load_yaml(config_path)
            failed_root = _resolve_root(config_path)
            failed_log = failed_root / str(failed_config.get("log_path", "logs/pipeline.jsonl"))
            JsonlLogger(failed_log, str(uuid.uuid4())).event(
                "command",
                "failed",
                exit_code=1,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        except Exception:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
