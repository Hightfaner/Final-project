from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


SPLIT_ORDER = ("train", "validation", "test")


class SplitError(RuntimeError):
    """Raised when an indivisible group split cannot meet the frozen contract."""


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.DataFrame
    summary: dict[str, Any]


def _largest_remainder(total: int, counts: dict[str, int]) -> dict[str, int]:
    overall = sum(counts.values())
    exact = {name: total * counts[name] / overall for name in SPLIT_ORDER}
    allocated = {name: math.floor(exact[name]) for name in SPLIT_ORDER}
    remaining = total - sum(allocated.values())
    ranked = sorted(SPLIT_ORDER, key=lambda name: (-(exact[name] - allocated[name]), name))
    for name in ranked[:remaining]:
        allocated[name] += 1
    return allocated


def _seed_order(seed: int, attempt: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{attempt}:{group_id}".encode("utf-8")).hexdigest()


def build_fixed_split(frame: pd.DataFrame, config: dict[str, Any]) -> SplitResult:
    required = {"email_id", "template_group_id", "label_binary"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SplitError(f"Split input is missing columns: {missing}")
    if not frame["email_id"].is_unique:
        raise SplitError("Split input email_id values must be unique")

    expected_counts = {name: int(config["expected_counts"][name]) for name in SPLIT_ORDER}
    if sum(expected_counts.values()) != len(frame):
        raise SplitError(
            f"Expected split counts sum to {sum(expected_counts.values())}, "
            f"but eligible rows total {len(frame)}"
        )
    total_phishing = int(frame["label_binary"].sum())
    desired_phishing = _largest_remainder(total_phishing, expected_counts)
    desired_legitimate = {
        name: expected_counts[name] - desired_phishing[name] for name in SPLIT_ORDER
    }

    groups = (
        frame.groupby("template_group_id", sort=True)
        .agg(
            size=("email_id", "size"),
            phishing=("label_binary", "sum"),
        )
        .reset_index()
    )
    groups["legitimate"] = groups["size"] - groups["phishing"]
    multi = groups.loc[groups["size"].gt(1)].copy()
    single = groups.loc[groups["size"].eq(1)].copy()
    seed = int(config.get("seed", 42))

    assignment_by_group: dict[str, str] | None = None
    for attempt in range(256):
        counts = {name: 0 for name in SPLIT_ORDER}
        phishing = {name: 0 for name in SPLIT_ORDER}
        legitimate = {name: 0 for name in SPLIT_ORDER}
        candidate_assignment: dict[str, str] = {}
        ordered = multi.copy()
        ordered["tie"] = ordered["template_group_id"].map(
            lambda value: _seed_order(seed, attempt, str(value))
        )
        ordered = ordered.sort_values(
            ["size", "phishing", "legitimate", "tie"],
            ascending=[False, False, False, True],
        )
        failed = False
        for row in ordered.itertuples(index=False):
            feasible: list[tuple[float, str]] = []
            for split_name in SPLIT_ORDER:
                new_count = counts[split_name] + int(row.size)
                new_phishing = phishing[split_name] + int(row.phishing)
                new_legitimate = legitimate[split_name] + int(row.legitimate)
                if new_count > expected_counts[split_name]:
                    continue
                if new_phishing > desired_phishing[split_name]:
                    continue
                if new_legitimate > desired_legitimate[split_name]:
                    continue
                remaining_total = expected_counts[split_name] - new_count
                remaining_phishing = desired_phishing[split_name] - new_phishing
                remaining_legitimate = desired_legitimate[split_name] - new_legitimate
                score = (
                    (remaining_total / max(1, expected_counts[split_name])) ** 2
                    + (remaining_phishing / max(1, desired_phishing[split_name])) ** 2
                    + (remaining_legitimate / max(1, desired_legitimate[split_name])) ** 2
                )
                feasible.append((score, split_name))
            if not feasible:
                failed = True
                break
            _, chosen = min(feasible, key=lambda item: (item[0], SPLIT_ORDER.index(item[1])))
            candidate_assignment[str(row.template_group_id)] = chosen
            counts[chosen] += int(row.size)
            phishing[chosen] += int(row.phishing)
            legitimate[chosen] += int(row.legitimate)
        if failed:
            continue

        phishing_singles = single.loc[single["phishing"].eq(1), "template_group_id"].astype(str).tolist()
        legitimate_singles = single.loc[single["legitimate"].eq(1), "template_group_id"].astype(str).tolist()
        phishing_singles.sort(key=lambda value: _seed_order(seed, attempt, value))
        legitimate_singles.sort(key=lambda value: _seed_order(seed + 1, attempt, value))
        phish_offset = 0
        legit_offset = 0
        for split_name in SPLIT_ORDER:
            need_phish = desired_phishing[split_name] - phishing[split_name]
            need_legit = desired_legitimate[split_name] - legitimate[split_name]
            for group_id in phishing_singles[phish_offset : phish_offset + need_phish]:
                candidate_assignment[group_id] = split_name
            for group_id in legitimate_singles[legit_offset : legit_offset + need_legit]:
                candidate_assignment[group_id] = split_name
            phish_offset += need_phish
            legit_offset += need_legit
        if phish_offset != len(phishing_singles) or legit_offset != len(legitimate_singles):
            continue
        if len(candidate_assignment) != len(groups):
            continue
        assignment_by_group = candidate_assignment
        break

    if assignment_by_group is None:
        raise SplitError("Unable to assign indivisible template groups within frozen count targets")

    assignments = frame[["email_id", "template_group_id", "label_binary"]].copy()
    assignments["split"] = assignments["template_group_id"].map(assignment_by_group)
    if assignments["split"].isna().any():
        raise SplitError("At least one template group was left unassigned")
    assignments = assignments.sort_values(["split", "email_id"]).reset_index(drop=True)

    actual_counts = assignments["split"].value_counts().to_dict()
    cross_split = int(assignments.groupby("template_group_id")["split"].nunique().gt(1).sum())
    if cross_split:
        raise SplitError(f"Found {cross_split} template groups spanning multiple splits")
    if any(actual_counts.get(name, 0) != expected_counts[name] for name in SPLIT_ORDER):
        raise SplitError(f"Split counts do not match frozen targets: {actual_counts}")

    overall_ratio = float(assignments["label_binary"].mean())
    class_ratios = assignments.groupby("split")["label_binary"].mean().to_dict()
    max_class_deviation = float(config.get("max_class_ratio_deviation", 0.03))
    for split_name, ratio in class_ratios.items():
        if abs(float(ratio) - overall_ratio) > max_class_deviation:
            raise SplitError(
                f"{split_name} phishing ratio deviation exceeds {max_class_deviation}: "
                f"{ratio} vs {overall_ratio}"
            )

    summary = {
        "seed": seed,
        "target_counts": expected_counts,
        "actual_counts": {name: int(actual_counts[name]) for name in SPLIT_ORDER},
        "target_phishing_counts": desired_phishing,
        "actual_phishing_counts": {
            name: int(assignments.loc[assignments["split"].eq(name), "label_binary"].sum())
            for name in SPLIT_ORDER
        },
        "overall_phishing_ratio": overall_ratio,
        "split_phishing_ratios": {name: float(class_ratios[name]) for name in SPLIT_ORDER},
        "template_group_count": int(assignments["template_group_id"].nunique()),
        "cross_split_template_group_violations": cross_split,
    }
    return SplitResult(
        assignments=assignments[["email_id", "split"]],
        summary=summary,
    )

