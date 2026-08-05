from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.io_utils import load_yaml, write_csv_atomic, write_text_atomic


RANDOM_SEED = 42
PARAMETER_GRID = {
    "max_depth": (2, 3, 4),
    "learning_rate": (0.05, 0.10),
    "n_estimators": (100, 200),
    "min_child_weight": (1, 5),
}
FIXED_PARAMETERS = {
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}
THRESHOLDS = tuple(round(0.30 + index * 0.01, 2) for index in range(41))
SEARCH_THRESHOLD = 0.50
MODEL_PATH = Path("models/xgboost_model.json")
FROZEN_CONFIG_PATH = Path("configs/frozen_model_config.json")
VALIDATION_RESULTS_PATH = Path("reports/model/validation_results.csv")
TEST_PREDICTIONS_PATH = Path("reports/model/test_predictions.csv")
MODEL_METRICS_PATH = Path("reports/model/model_metrics.json")


class ModelTrainingError(RuntimeError):
    """Raised when the frozen modelling contract cannot be satisfied."""


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _candidate_parameters() -> list[dict[str, int | float | str]]:
    candidates = []
    for max_depth, learning_rate, n_estimators, min_child_weight in itertools.product(
        PARAMETER_GRID["max_depth"],
        PARAMETER_GRID["learning_rate"],
        PARAMETER_GRID["n_estimators"],
        PARAMETER_GRID["min_child_weight"],
    ):
        candidates.append(
            {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "min_child_weight": min_child_weight,
                **FIXED_PARAMETERS,
            }
        )
    return candidates


def _classifier(parameters: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        **parameters,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
    )


def _simple_model_key(result: dict[str, Any]) -> tuple[float, int, int, int, float]:
    return (
        -float(result["balanced_accuracy"]),
        int(result["max_depth"]),
        int(result["n_estimators"]),
        -int(result["min_child_weight"]),
        float(result["learning_rate"]),
    )


def _select_hyperparameters(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ModelTrainingError("Hyperparameter search produced no results")
    best_score = max(float(result["balanced_accuracy"]) for result in results)
    tied = [
        result
        for result in results
        if np.isclose(float(result["balanced_accuracy"]), best_score, rtol=0.0, atol=1e-12)
    ]
    return min(tied, key=_simple_model_key)


def _select_threshold(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ModelTrainingError("Threshold search produced no results")
    best_score = max(float(result["balanced_accuracy"]) for result in results)
    tied = [
        result
        for result in results
        if np.isclose(float(result["balanced_accuracy"]), best_score, rtol=0.0, atol=1e-12)
    ]
    return min(
        tied,
        key=lambda result: (
            abs(float(result["threshold"]) - 0.50),
            result["threshold"],
        ),
    )


def _result_row(
    *,
    stage: str,
    candidate_id: str,
    parameters: dict[str, Any],
    threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "candidate_id": candidate_id,
        "max_depth": int(parameters["max_depth"]),
        "learning_rate": float(parameters["learning_rate"]),
        "n_estimators": int(parameters["n_estimators"]),
        "min_child_weight": int(parameters["min_child_weight"]),
        "subsample": float(parameters["subsample"]),
        "colsample_bytree": float(parameters["colsample_bytree"]),
        "objective": str(parameters["objective"]),
        "eval_metric": str(parameters["eval_metric"]),
        "threshold": float(threshold),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
        "selected_hyperparameters": False,
        "selected_threshold": False,
    }


def _validate_data_contract(
    frame: pd.DataFrame,
    feature_order: list[str],
) -> pd.DataFrame:
    if len(feature_order) != 8 or len(set(feature_order)) != 8:
        raise ModelTrainingError("Configuration must contain exactly eight unique features")
    required = {"email_id", "label", "split", *feature_order}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ModelTrainingError(f"Processed data is missing columns: {missing}")
    if set(frame["label"]) != {0, 1}:
        raise ModelTrainingError("label must contain exactly 0 and 1")
    if set(frame["split"]) != {"train", "validation", "test"}:
        raise ModelTrainingError("split must contain train, validation, and test")
    if frame["email_id"].duplicated().any():
        raise ModelTrainingError("email_id must be unique")

    numeric = frame[feature_order].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ModelTrainingError("Features contain NaN or Inf")
    validated = frame.copy()
    validated[feature_order] = numeric
    return validated


def _save_model_atomic(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".json",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        model.save_model(temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _outcome_types(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y_true == 1) & (predictions == 1),
            (y_true == 0) & (predictions == 0),
            (y_true == 0) & (predictions == 1),
            (y_true == 1) & (predictions == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="",
    )


def _availability(test_metrics: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "TP": int(test_metrics["tp"]),
        "TN": int(test_metrics["tn"]),
        "FP": int(test_metrics["fp"]),
        "FN": int(test_metrics["fn"]),
    }
    by_type = {
        outcome: {
            "count": count,
            "at_least_4": count >= 4,
            "at_least_6": count >= 6,
        }
        for outcome, count in counts.items()
    }
    return {
        "by_outcome_type": by_type,
        "all_types_at_least_4": all(value["at_least_4"] for value in by_type.values()),
        "all_types_at_least_6": all(value["at_least_6"] for value in by_type.values()),
    }


def train_and_evaluate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    processed_path = root / str(config["output"]["processed_csv"])
    frame = pd.read_csv(processed_path, keep_default_na=False)
    feature_order = [str(value) for value in config["feature_order"]]
    frame = _validate_data_contract(frame, feature_order)

    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    x_train = train.loc[:, feature_order]
    y_train = train["label"].to_numpy(dtype=int)
    x_validation = validation.loc[:, feature_order]
    y_validation = validation["label"].to_numpy(dtype=int)

    search_rows: list[dict[str, Any]] = []
    hyperparameter_results: list[dict[str, Any]] = []
    candidates = _candidate_parameters()
    for index, parameters in enumerate(candidates, start=1):
        model = _classifier(parameters)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        metrics = _metrics(y_validation, probabilities, SEARCH_THRESHOLD)
        row = _result_row(
            stage="hyperparameter_search",
            candidate_id=f"model_{index:02d}",
            parameters=parameters,
            threshold=SEARCH_THRESHOLD,
            metrics=metrics,
        )
        search_rows.append(row)
        hyperparameter_results.append(row)

    selected_hyperparameters_row = _select_hyperparameters(hyperparameter_results)
    selected_hyperparameters_row["selected_hyperparameters"] = True
    selected_parameters = {
        name: selected_hyperparameters_row[name]
        for name in (
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "objective",
            "eval_metric",
        )
    }

    final_model = _classifier(selected_parameters)
    final_model.fit(x_train, y_train)
    validation_probabilities = final_model.predict_proba(x_validation)[:, 1]
    threshold_results: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        metrics = _metrics(y_validation, validation_probabilities, threshold)
        row = _result_row(
            stage="threshold_search",
            candidate_id=f"threshold_{threshold:.2f}",
            parameters=selected_parameters,
            threshold=threshold,
            metrics=metrics,
        )
        row["selected_hyperparameters"] = True
        search_rows.append(row)
        threshold_results.append(row)

    selected_threshold_row = _select_threshold(threshold_results)
    selected_threshold_row["selected_threshold"] = True
    selected_threshold = float(selected_threshold_row["threshold"])
    validation_metrics = _metrics(
        y_validation,
        validation_probabilities,
        selected_threshold,
    )

    model_path = root / MODEL_PATH
    _save_model_atomic(final_model, model_path)
    reloaded_model = XGBClassifier()
    reloaded_model.load_model(model_path)
    reloaded_validation_probabilities = reloaded_model.predict_proba(x_validation)[:, 1]
    reload_probabilities_match = bool(
        np.allclose(
            validation_probabilities,
            reloaded_validation_probabilities,
            rtol=0.0,
            atol=1e-12,
        )
    )
    reload_labels_match = bool(
        np.array_equal(
            validation_probabilities >= selected_threshold,
            reloaded_validation_probabilities >= selected_threshold,
        )
    )
    if not reload_probabilities_match or not reload_labels_match:
        raise ModelTrainingError("Reloaded model changed validation predictions")

    # Test is first predicted here, after model and threshold selection and reload validation.
    x_test = test.loc[:, feature_order]
    y_test = test["label"].to_numpy(dtype=int)
    test_probabilities = reloaded_model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= selected_threshold).astype(int)
    test_metrics = _metrics(y_test, test_probabilities, selected_threshold)
    outcome_types = _outcome_types(y_test, test_predictions)
    test_prediction_frame = pd.DataFrame(
        {
            "email_id": test["email_id"].astype(str).to_numpy(),
            "true_label": y_test,
            "phishing_probability": test_probabilities,
            "predicted_label": test_predictions,
            "outcome_type": outcome_types,
        }
    )
    availability = _availability(test_metrics)

    frozen_config = {
        "feature_order": feature_order,
        "label_definition": {"0": "legitimate", "1": "phishing"},
        "hyperparameters": selected_parameters,
        "classification_threshold": selected_threshold,
        "random_seed": RANDOM_SEED,
    }
    metrics_document = {
        "selection": {
            "hyperparameter_selection_metric": "validation_balanced_accuracy",
            "hyperparameter_selection_threshold": SEARCH_THRESHOLD,
            "hyperparameter_candidate_count": len(candidates),
            "hyperparameter_tie_breaking": [
                "lower max_depth",
                "lower n_estimators",
                "higher min_child_weight",
                "lower learning_rate",
            ],
            "threshold_selection_metric": "validation_balanced_accuracy",
            "threshold_start": THRESHOLDS[0],
            "threshold_end": THRESHOLDS[-1],
            "threshold_step": 0.01,
            "threshold_tie_breaking": "closest to 0.50, then lower threshold",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_outcome_availability": availability,
        "reload_prediction_consistency": {
            "probabilities_match": reload_probabilities_match,
            "labels_match": reload_labels_match,
        },
        "row_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "processed_csv_sha256": hashlib.sha256(processed_path.read_bytes()).hexdigest(),
    }

    validation_results = pd.DataFrame(search_rows)
    validation_results = validation_results[
        [
            "stage",
            "candidate_id",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "objective",
            "eval_metric",
            "threshold",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "tn",
            "fp",
            "fn",
            "tp",
            "selected_hyperparameters",
            "selected_threshold",
        ]
    ]
    write_csv_atomic(root / VALIDATION_RESULTS_PATH, validation_results)
    write_csv_atomic(root / TEST_PREDICTIONS_PATH, test_prediction_frame)
    write_text_atomic(
        root / FROZEN_CONFIG_PATH,
        json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(
        root / MODEL_METRICS_PATH,
        json.dumps(metrics_document, ensure_ascii=False, indent=2) + "\n",
    )

    return {
        "status": "PASS",
        "selected_hyperparameters": selected_parameters,
        "selected_threshold": selected_threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_outcome_availability": availability,
        "reload_prediction_consistency": metrics_document[
            "reload_prediction_consistency"
        ],
        "artifacts": {
            "model": str(model_path),
            "frozen_config": str(root / FROZEN_CONFIG_PATH),
            "validation_results": str(root / VALIDATION_RESULTS_PATH),
            "test_predictions": str(root / TEST_PREDICTIONS_PATH),
            "model_metrics": str(root / MODEL_METRICS_PATH),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and freeze the XGBoost phishing model")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_yaml(config_path)
        root = config_path.parent.parent
        result = train_and_evaluate(root, config)
    except (ModelTrainingError, KeyError, TypeError, ValueError) as error:
        print(f"Model training failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
