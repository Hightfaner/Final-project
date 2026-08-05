from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from xgboost import DMatrix, XGBClassifier

from src.io_utils import write_text_atomic


DEFAULT_MODEL_PATH = Path("models/xgboost_model.json")
DEFAULT_FROZEN_CONFIG_PATH = Path("configs/frozen_model_config.json")
DEFAULT_PROCESSED_PATH = Path("data/processed/processed_emails.csv")
DEFAULT_PREDICTIONS_PATH = Path("reports/model/test_predictions.csv")
DEFAULT_OUTPUT_PATH = Path("reports/model/treeshap_explanations.json")
PREDICTION_TOLERANCE = 1e-7
ADDITIVITY_TOLERANCE = 1e-5

DISPLAY_NAMES = {
    "url_count": "Number of links",
    "ip_address_url_count": "IP-address links",
    "urgency_word_count": "Urgency-related words",
    "credential_word_count": "Account and credential words",
    "action_word_count": "Action-request words",
    "money_related_word_count": "Money-related words",
    "uppercase_letter_ratio": "Uppercase-letter proportion",
    "exclamation_mark_count": "Exclamation marks",
}
COUNT_FEATURES = set(DISPLAY_NAMES) - {"uppercase_letter_ratio"}
FORBIDDEN_JSON_KEYS = {
    "true_label",
    "label",
    "shap_value",
    "shap_values",
    "raw_margin",
    "base_value",
}


class TreeShapGenerationError(RuntimeError):
    """Raised when a frozen modelling or explanation contract is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TreeShapGenerationError(f"Required file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TreeShapGenerationError(f"Expected a JSON object in {path}")
    return value


def _validate_frozen_config(config: dict[str, Any]) -> tuple[list[str], float]:
    feature_order = [str(value) for value in config.get("feature_order", [])]
    if len(feature_order) != 8 or len(set(feature_order)) != 8:
        raise TreeShapGenerationError("Frozen config must contain eight unique features")
    if feature_order != list(DISPLAY_NAMES):
        raise TreeShapGenerationError(
            "Frozen feature names or order do not match the explanation display mapping"
        )
    labels = config.get("label_definition")
    if labels != {"0": "legitimate", "1": "phishing"}:
        raise TreeShapGenerationError("Frozen label mapping must be 0=legitimate, 1=phishing")
    threshold = float(config.get("classification_threshold"))
    if not 0.0 <= threshold <= 1.0:
        raise TreeShapGenerationError("Frozen classification threshold is invalid")
    return feature_order, threshold


def _validate_inputs(
    processed: pd.DataFrame,
    stored_predictions: pd.DataFrame,
    feature_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_required = {"email_id", "split", *feature_order}
    missing_processed = sorted(processed_required - set(processed.columns))
    if missing_processed:
        raise TreeShapGenerationError(
            f"Processed data is missing required columns: {missing_processed}"
        )
    prediction_required = {
        "email_id",
        "phishing_probability",
        "predicted_label",
    }
    missing_predictions = sorted(prediction_required - set(stored_predictions.columns))
    if missing_predictions:
        raise TreeShapGenerationError(
            f"Stored predictions are missing required columns: {missing_predictions}"
        )

    test = processed.loc[processed["split"].eq("test")].copy()
    predictions = stored_predictions.copy()
    if test["email_id"].duplicated().any() or predictions["email_id"].duplicated().any():
        raise TreeShapGenerationError("Test and prediction email_id values must be unique")
    test_ids = test["email_id"].astype(str).tolist()
    prediction_ids = predictions["email_id"].astype(str).tolist()
    if test_ids != prediction_ids:
        raise TreeShapGenerationError(
            "Test email_id values or order do not match the frozen predictions"
        )

    numeric_features = test[feature_order].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise TreeShapGenerationError("Test features contain NaN or Inf")
    test[feature_order] = numeric_features
    predictions["phishing_probability"] = pd.to_numeric(
        predictions["phishing_probability"],
        errors="raise",
    )
    predictions["predicted_label"] = pd.to_numeric(
        predictions["predicted_label"],
        errors="raise",
    ).astype(int)
    return test, predictions


def _format_observed_value(feature_key: str, value: float) -> str:
    if feature_key == "uppercase_letter_ratio":
        percentage = f"{value * 100:.1f}".rstrip("0").rstrip(".")
        return f"{percentage}%"
    if not np.isclose(value, round(value), rtol=0.0, atol=1e-12):
        raise TreeShapGenerationError(f"Count feature is not an integer: {feature_key}")
    return str(int(round(value)))


def _json_observed_value(feature_key: str, value: float) -> int | float:
    if feature_key in COUNT_FEATURES:
        return int(round(value))
    return float(value)


def _direction(shap_value: float) -> str:
    if shap_value > 0:
        return "pushed the prediction towards phishing"
    if shap_value < 0:
        return "pushed the prediction towards legitimate"
    raise TreeShapGenerationError("A selected Top 3 feature has exactly zero SHAP value")


def _top_feature_indices(shap_row: np.ndarray) -> list[int]:
    return sorted(
        range(len(shap_row)),
        key=lambda index: (-abs(float(shap_row[index])), index),
    )[:3]


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_JSON_KEYS.intersection(value)
        if forbidden:
            raise TreeShapGenerationError(
                f"Participant JSON contains forbidden keys: {sorted(forbidden)}"
            )
        for nested in value.values():
            _validate_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_forbidden_keys(nested)


def generate_explanations(
    *,
    model_path: Path,
    frozen_config_path: Path,
    processed_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_config = _load_json(frozen_config_path)
    feature_order, threshold = _validate_frozen_config(frozen_config)
    if not model_path.is_file():
        raise TreeShapGenerationError(f"Frozen model does not exist: {model_path}")
    if not processed_path.is_file() or not predictions_path.is_file():
        raise TreeShapGenerationError("Processed test data or frozen predictions are missing")

    processed = pd.read_csv(processed_path, keep_default_na=False)
    stored_predictions = pd.read_csv(predictions_path, keep_default_na=False)
    test, stored_predictions = _validate_inputs(
        processed,
        stored_predictions,
        feature_order,
    )
    x_test = test.loc[:, feature_order]

    model = XGBClassifier()
    model.load_model(model_path)
    booster = model.get_booster()
    if booster.feature_names != feature_order:
        raise TreeShapGenerationError(
            "Frozen model feature names or order do not match the frozen config"
        )

    model_probabilities = model.predict_proba(x_test)[:, 1]
    stored_probabilities = stored_predictions["phishing_probability"].to_numpy(dtype=float)
    maximum_probability_difference = float(
        np.max(np.abs(model_probabilities.astype(float) - stored_probabilities))
    )
    if maximum_probability_difference > PREDICTION_TOLERANCE:
        raise TreeShapGenerationError(
            "Reloaded model probabilities do not match the frozen test predictions"
        )
    model_labels = (model_probabilities >= threshold).astype(int)
    stored_labels = stored_predictions["predicted_label"].to_numpy(dtype=int)
    if not np.array_equal(model_labels, stored_labels):
        raise TreeShapGenerationError(
            "Reloaded model labels do not match the frozen test predictions"
        )

    explainer = shap.TreeExplainer(booster, model_output="raw")
    shap_values = np.asarray(
        explainer.shap_values(x_test, check_additivity=True),
        dtype=float,
    )
    if shap_values.shape != (len(test), len(feature_order)):
        raise TreeShapGenerationError(
            f"Unexpected SHAP matrix shape: {shap_values.shape}"
        )
    expected_values = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if len(expected_values) != 1:
        raise TreeShapGenerationError("Expected one scalar TreeSHAP base value")
    raw_margins = booster.predict(
        DMatrix(x_test, feature_names=feature_order),
        output_margin=True,
    )
    additive_reconstruction = expected_values[0] + shap_values.sum(axis=1)
    maximum_additivity_error = float(
        np.max(np.abs(additive_reconstruction - raw_margins))
    )
    if maximum_additivity_error > ADDITIVITY_TOLERANCE:
        raise TreeShapGenerationError(
            "TreeSHAP values do not reconstruct the frozen model raw margin"
        )

    explanations: list[dict[str, Any]] = []
    for row_index, (_, test_row) in enumerate(test.iterrows()):
        shap_row = shap_values[row_index]
        top_indices = _top_feature_indices(shap_row)
        if any(
            abs(shap_row[top_indices[index]])
            < abs(shap_row[top_indices[index + 1]])
            for index in range(len(top_indices) - 1)
        ):
            raise TreeShapGenerationError("Top 3 features are not sorted by absolute SHAP value")

        features = []
        for rank, feature_index in enumerate(top_indices, start=1):
            feature_key = feature_order[feature_index]
            observed_value = float(test_row[feature_key])
            display_name = DISPLAY_NAMES[feature_key]
            direction = _direction(float(shap_row[feature_index]))
            formatted_value = _format_observed_value(feature_key, observed_value)
            features.append(
                {
                    "rank": rank,
                    "feature_key": feature_key,
                    "display_name": display_name,
                    "observed_value": _json_observed_value(
                        feature_key,
                        observed_value,
                    ),
                    "direction": direction,
                    "display_text": (
                        f"{display_name}: {formatted_value} — {direction}"
                    ),
                }
            )
        explanations.append(
            {
                "email_id": str(test_row["email_id"]),
                "ai_prediction": "Phishing" if model_labels[row_index] == 1 else "Legitimate",
                "features": features,
            }
        )

    document = {"explanations": explanations}
    _validate_no_forbidden_keys(document)
    summary = {
        "status": "PASS",
        "explanation_count": len(explanations),
        "feature_order_matches": True,
        "frozen_prediction_labels_match": True,
        "maximum_probability_difference": maximum_probability_difference,
        "maximum_additivity_error": maximum_additivity_error,
        "top_three_sorted": True,
        "participant_json_has_no_forbidden_keys": True,
    }
    return document, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline participant-facing TreeSHAP explanations"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--frozen-config",
        type=Path,
        default=DEFAULT_FROZEN_CONFIG_PATH,
    )
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document, summary = generate_explanations(
            model_path=args.model.resolve(),
            frozen_config_path=args.frozen_config.resolve(),
            processed_path=args.processed.resolve(),
            predictions_path=args.predictions.resolve(),
        )
        output_path = args.output.resolve()
        write_text_atomic(
            output_path,
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        )
        summary["output"] = str(output_path)
    except (
        TreeShapGenerationError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"TreeSHAP generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
