from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import kmedoids
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.io_utils import write_text_atomic
from src.treeshap_explanations import DISPLAY_NAMES, _format_observed_value


DEFAULT_MODEL_PATH = Path("models/xgboost_model.json")
DEFAULT_FROZEN_CONFIG_PATH = Path("configs/frozen_model_config.json")
DEFAULT_PROCESSED_PATH = Path("data/processed/processed_emails.csv")
DEFAULT_PREDICTIONS_PATH = Path("reports/model/test_predictions.csv")
DEFAULT_OUTPUT_PATH = Path("reports/model/kmedoids_explanations.json")
RANDOM_SEED = 42
K_VALUES = (2, 3, 4, 5, 6)
SHARED_FEATURE_MAX_DIFFERENCE = 0.5
DISTANCE_PERCENTILE = 95
PREDICTION_TOLERANCE = 1e-7
RATIO_FEATURE_KEY = "uppercase_letter_ratio"
FORBIDDEN_JSON_KEYS = {
    "true_label",
    "label",
    "outcome_type",
    "cluster_label",
    "scaled_value",
    "scaled_features",
    "distance",
    "distance_threshold",
    "silhouette_score",
    "phishing_probability",
    "probability",
    "shap_value",
    "shap_values",
    "raw_margin",
    "base_value",
}


class KMedoidsGenerationError(RuntimeError):
    """Raised when a frozen input or example-explanation rule is violated."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise KMedoidsGenerationError(f"Required file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise KMedoidsGenerationError(f"Expected a JSON object in {path}")
    return value


def _validate_frozen_config(config: dict[str, Any]) -> tuple[list[str], float]:
    feature_order = [str(value) for value in config.get("feature_order", [])]
    if feature_order != list(DISPLAY_NAMES) or len(feature_order) != 8:
        raise KMedoidsGenerationError(
            "Frozen feature names or order do not match the TreeSHAP display mapping"
        )
    if config.get("label_definition") != {"0": "legitimate", "1": "phishing"}:
        raise KMedoidsGenerationError("Frozen labels must be 0=legitimate, 1=phishing")
    threshold = float(config.get("classification_threshold"))
    if not np.isclose(threshold, 0.35, rtol=0.0, atol=1e-12):
        raise KMedoidsGenerationError("Frozen classification threshold is not 0.35")
    if int(config.get("random_seed")) != RANDOM_SEED:
        raise KMedoidsGenerationError("Frozen random seed is not 42")
    return feature_order, threshold


def _validate_data(
    processed: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    display_columns = ["email_id", "split", "sanitized_subject", "sanitized_body"]
    required_processed = {*display_columns, *feature_order}
    missing_processed = sorted(required_processed - set(processed.columns))
    if missing_processed:
        raise KMedoidsGenerationError(
            f"Processed data is missing required columns: {missing_processed}"
        )
    required_predictions = {
        "email_id",
        "phishing_probability",
        "predicted_label",
        "outcome_type",
    }
    missing_predictions = sorted(required_predictions - set(predictions.columns))
    if missing_predictions:
        raise KMedoidsGenerationError(
            f"Stored predictions are missing required columns: {missing_predictions}"
        )

    modelling_columns = display_columns + feature_order
    model_data = processed.loc[:, modelling_columns].copy()
    model_data[feature_order] = model_data[feature_order].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(model_data[feature_order].to_numpy(dtype=float)).all():
        raise KMedoidsGenerationError("Frozen features contain NaN or Inf")
    train = model_data.loc[model_data["split"].eq("train")].copy()
    test = model_data.loc[model_data["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise KMedoidsGenerationError("Frozen train or test split is empty")
    if train["email_id"].duplicated().any() or test["email_id"].duplicated().any():
        raise KMedoidsGenerationError("Train and test email_id values must be unique")
    if set(train["email_id"].astype(str)).intersection(test["email_id"].astype(str)):
        raise KMedoidsGenerationError("Train and test email_id values overlap")

    frozen_predictions = predictions.loc[
        :, ["email_id", "phishing_probability", "predicted_label", "outcome_type"]
    ].copy()
    frozen_predictions["phishing_probability"] = pd.to_numeric(
        frozen_predictions["phishing_probability"],
        errors="raise",
    )
    frozen_predictions["predicted_label"] = pd.to_numeric(
        frozen_predictions["predicted_label"],
        errors="raise",
    ).astype(int)
    if test["email_id"].astype(str).tolist() != frozen_predictions["email_id"].astype(str).tolist():
        raise KMedoidsGenerationError(
            "Test email_id values or order do not match the frozen predictions"
        )
    return train, test, frozen_predictions


def _fit_class_medoids(
    *,
    train_scaled: np.ndarray,
    train_predictions: np.ndarray,
    train_ids: np.ndarray,
    predicted_label: int,
) -> dict[str, Any]:
    class_indices = np.flatnonzero(train_predictions == predicted_label)
    if len(class_indices) <= max(K_VALUES):
        raise KMedoidsGenerationError(
            f"Too few train samples for predicted class {predicted_label}"
        )
    class_scaled = train_scaled[class_indices]
    dissimilarities = pairwise_distances(class_scaled, metric="euclidean")
    candidate_results: list[dict[str, Any]] = []

    for k in K_VALUES:
        clusterer = kmedoids.KMedoids(
            n_clusters=k,
            metric="precomputed",
            method="pam",
            init="build",
            max_iter=300,
            random_state=RANDOM_SEED,
        )
        clusterer.fit(dissimilarities)
        labels = np.asarray(clusterer.labels_, dtype=int)
        local_medoid_indices = np.asarray(clusterer.medoid_indices_, dtype=int)
        if len(np.unique(labels)) != k or len(local_medoid_indices) != k:
            raise KMedoidsGenerationError(f"k-medoids returned invalid k={k} clustering")
        score = float(silhouette_score(class_scaled, labels, metric="euclidean"))
        candidate_results.append(
            {
                "k": k,
                "silhouette_score": score,
                "local_medoid_indices": local_medoid_indices,
            }
        )

    best_score = max(result["silhouette_score"] for result in candidate_results)
    selected = min(
        (
            result
            for result in candidate_results
            if np.isclose(
                result["silhouette_score"],
                best_score,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        key=lambda result: result["k"],
    )
    global_medoid_indices = class_indices[selected["local_medoid_indices"]]
    global_medoid_indices = np.asarray(
        sorted(global_medoid_indices, key=lambda index: str(train_ids[index])),
        dtype=int,
    )
    medoid_vectors = train_scaled[global_medoid_indices]
    train_to_medoids = pairwise_distances(
        class_scaled,
        medoid_vectors,
        metric="euclidean",
    )
    nearest_train_distances = train_to_medoids.min(axis=1)
    distance_threshold = float(
        np.percentile(
            nearest_train_distances,
            DISTANCE_PERCENTILE,
            method="linear",
        )
    )
    return {
        "predicted_label": predicted_label,
        "train_count": len(class_indices),
        "k": int(selected["k"]),
        "silhouette_score": float(selected["silhouette_score"]),
        "candidate_scores": {
            str(result["k"]): float(result["silhouette_score"])
            for result in candidate_results
        },
        "global_medoid_indices": global_medoid_indices,
        "medoid_ids": [str(train_ids[index]) for index in global_medoid_indices],
        "distance_threshold": distance_threshold,
    }


def _nearest_medoid(
    test_vector: np.ndarray,
    train_scaled: np.ndarray,
    train_ids: np.ndarray,
    medoid_indices: np.ndarray,
) -> tuple[int, float]:
    distances = np.linalg.norm(train_scaled[medoid_indices] - test_vector, axis=1)
    ordered = sorted(
        range(len(medoid_indices)),
        key=lambda position: (
            float(distances[position]),
            str(train_ids[medoid_indices[position]]),
        ),
    )
    selected_position = ordered[0]
    selected_index = int(medoid_indices[selected_position])
    selected_distance = float(distances[selected_position])
    if selected_distance != float(np.min(distances)):
        raise KMedoidsGenerationError("Selected representative is not the nearest medoid")
    tied_ids = sorted(
        str(train_ids[medoid_indices[position]])
        for position in range(len(medoid_indices))
        if float(distances[position]) == selected_distance
    )
    if str(train_ids[selected_index]) != tied_ids[0]:
        raise KMedoidsGenerationError("Equal-distance medoid tie-break is not stable")
    return selected_index, selected_distance


def _json_feature_value(feature_key: str, value: float) -> int | float:
    if feature_key == "uppercase_letter_ratio":
        return float(value)
    if not np.isclose(value, round(value), rtol=0.0, atol=1e-12):
        raise KMedoidsGenerationError(f"Count feature is not an integer: {feature_key}")
    return int(round(value))


def _shared_features(
    *,
    feature_order: list[str],
    test_raw: np.ndarray,
    example_raw: np.ndarray,
    test_scaled: np.ndarray,
    example_scaled: np.ndarray,
    constant_mask: np.ndarray,
) -> list[dict[str, Any]]:
    standardised_differences = np.abs(test_scaled - example_scaled)
    candidates: list[tuple[float, int]] = []
    for feature_index, difference in enumerate(standardised_differences):
        if float(difference) > SHARED_FEATURE_MAX_DIFFERENCE:
            continue
        if (
            constant_mask[feature_index]
            and test_raw[feature_index] != example_raw[feature_index]
        ):
            continue
        feature_key = feature_order[feature_index]
        if feature_key != RATIO_FEATURE_KEY:
            current_value = float(test_raw[feature_index])
            example_value = float(example_raw[feature_index])
            if current_value <= 0.0 or example_value <= 0.0:
                continue
            if abs(current_value - example_value) > 1.0:
                continue
        candidates.append((float(difference), feature_index))
    candidates.sort(key=lambda item: (item[0], item[1]))

    shared = []
    for rank, (_, feature_index) in enumerate(candidates[:3], start=1):
        feature_key = feature_order[feature_index]
        display_name = DISPLAY_NAMES[feature_key]
        current_value = float(test_raw[feature_index])
        example_value = float(example_raw[feature_index])
        current_display = _format_observed_value(feature_key, current_value)
        example_display = _format_observed_value(feature_key, example_value)
        shared.append(
            {
                "rank": rank,
                "feature_key": feature_key,
                "display_name": display_name,
                "current_value": _json_feature_value(feature_key, current_value),
                "example_value": _json_feature_value(feature_key, example_value),
                "display_text": (
                    f"{display_name} — this email: {current_display}; "
                    f"representative email: {example_display}"
                ),
            }
        )
    return shared


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_JSON_KEYS.intersection(value)
        if forbidden:
            raise KMedoidsGenerationError(
                f"Participant JSON contains forbidden keys: {sorted(forbidden)}"
            )
        for nested in value.values():
            _validate_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_no_forbidden_keys(nested)


def _ineligible_record(email_id: str, prediction_name: str) -> dict[str, Any]:
    return {
        "email_id": email_id,
        "ai_prediction": prediction_name,
        "eligible": False,
        "representative_example": None,
        "shared_features": [],
        "display_intro": None,
    }


def generate_kmedoids_explanations(
    *,
    model_path: Path,
    frozen_config_path: Path,
    processed_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_config = _load_json(frozen_config_path)
    feature_order, threshold = _validate_frozen_config(frozen_config)
    if not model_path.is_file():
        raise KMedoidsGenerationError(f"Frozen model does not exist: {model_path}")
    if not processed_path.is_file() or not predictions_path.is_file():
        raise KMedoidsGenerationError("Processed data or frozen predictions are missing")

    processed = pd.read_csv(processed_path, keep_default_na=False)
    predictions = pd.read_csv(predictions_path, keep_default_na=False)
    train, test, frozen_predictions = _validate_data(
        processed,
        predictions,
        feature_order,
    )
    x_train = train.loc[:, feature_order]
    x_test = test.loc[:, feature_order]

    model = XGBClassifier()
    model.load_model(model_path)
    if model.get_booster().feature_names != feature_order:
        raise KMedoidsGenerationError(
            "Frozen model feature names or order do not match the frozen config"
        )
    train_probabilities = model.predict_proba(x_train)[:, 1]
    train_predictions = (train_probabilities >= threshold).astype(int)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    stored_probabilities = frozen_predictions["phishing_probability"].to_numpy(dtype=float)
    maximum_probability_difference = float(
        np.max(np.abs(test_probabilities.astype(float) - stored_probabilities))
    )
    if maximum_probability_difference > PREDICTION_TOLERANCE:
        raise KMedoidsGenerationError(
            "Reloaded model probabilities do not match frozen test predictions"
        )
    test_predictions = (test_probabilities >= threshold).astype(int)
    stored_test_predictions = frozen_predictions["predicted_label"].to_numpy(dtype=int)
    if not np.array_equal(test_predictions, stored_test_predictions):
        raise KMedoidsGenerationError(
            "Reloaded model labels do not match frozen test predictions"
        )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    if int(scaler.n_samples_seen_) != len(train):
        raise KMedoidsGenerationError("StandardScaler was not fitted on train only")
    test_scaled = scaler.transform(x_test)
    constant_mask = np.asarray(scaler.var_ == 0.0, dtype=bool)
    train_ids = train["email_id"].astype(str).to_numpy()

    class_models = {
        predicted_label: _fit_class_medoids(
            train_scaled=train_scaled,
            train_predictions=train_predictions,
            train_ids=train_ids,
            predicted_label=predicted_label,
        )
        for predicted_label in (0, 1)
    }
    all_medoid_indices = {
        int(index)
        for class_model in class_models.values()
        for index in class_model["global_medoid_indices"]
    }
    if not all(0 <= index < len(train) for index in all_medoid_indices):
        raise KMedoidsGenerationError("A medoid is not a real train row")

    explanations: list[dict[str, Any]] = []
    ineligible_distance = 0
    ineligible_shared_features = 0
    ineligible_missing_content = 0
    eligibility_by_id: dict[str, bool] = {}

    for test_position, (_, test_row) in enumerate(test.iterrows()):
        predicted_label = int(test_predictions[test_position])
        prediction_name = "phishing" if predicted_label == 1 else "legitimate"
        class_model = class_models[predicted_label]
        medoid_indices = class_model["global_medoid_indices"]
        medoid_index, distance = _nearest_medoid(
            test_scaled[test_position],
            train_scaled,
            train_ids,
            medoid_indices,
        )
        email_id = str(test_row["email_id"])
        if train_predictions[medoid_index] != predicted_label:
            raise KMedoidsGenerationError(
                "Representative AI prediction does not match the test prediction"
            )
        if distance > class_model["distance_threshold"]:
            explanations.append(_ineligible_record(email_id, prediction_name))
            eligibility_by_id[email_id] = False
            ineligible_distance += 1
            continue

        representative = train.iloc[medoid_index]
        subject = str(representative["sanitized_subject"])
        body = str(representative["sanitized_body"])
        if not subject.strip() and not body.strip():
            explanations.append(_ineligible_record(email_id, prediction_name))
            eligibility_by_id[email_id] = False
            ineligible_missing_content += 1
            continue

        shared_features = _shared_features(
            feature_order=feature_order,
            test_raw=x_test.iloc[test_position].to_numpy(dtype=float),
            example_raw=x_train.iloc[medoid_index].to_numpy(dtype=float),
            test_scaled=test_scaled[test_position],
            example_scaled=train_scaled[medoid_index],
            constant_mask=constant_mask,
        )
        if len(shared_features) < 2:
            explanations.append(_ineligible_record(email_id, prediction_name))
            eligibility_by_id[email_id] = False
            ineligible_shared_features += 1
            continue
        if not 2 <= len(shared_features) <= 3:
            raise KMedoidsGenerationError("Eligible explanation does not have 2-3 features")

        representative_id = str(representative["email_id"])
        if representative_id not in set(train_ids):
            raise KMedoidsGenerationError("Representative email_id is not in train")
        intro = (
            "This email is similar to a representative email that the AI also "
            f"classified as {prediction_name}."
        )
        explanations.append(
            {
                "email_id": email_id,
                "ai_prediction": prediction_name,
                "eligible": True,
                "representative_example": {
                    "email_id": representative_id,
                    "ai_prediction": prediction_name,
                    "subject": subject,
                    "body": body,
                },
                "shared_features": shared_features,
                "display_intro": intro,
            }
        )
        eligibility_by_id[email_id] = True

    document = {"explanations": explanations}
    _validate_no_forbidden_keys(document)
    zero_zero_violations = sum(
        1
        for explanation in explanations
        if explanation["eligible"]
        for feature in explanation["shared_features"]
        if feature["feature_key"] != RATIO_FEATURE_KEY
        if feature["current_value"] == 0 and feature["example_value"] == 0
    )
    count_feature_rule_violations = sum(
        1
        for explanation in explanations
        if explanation["eligible"]
        for feature in explanation["shared_features"]
        if feature["feature_key"] != RATIO_FEATURE_KEY
        if (
            feature["current_value"] <= 0
            or feature["example_value"] <= 0
            or abs(feature["current_value"] - feature["example_value"]) > 1
        )
    )
    if zero_zero_violations or count_feature_rule_violations:
        raise KMedoidsGenerationError(
            "Eligible explanations contain a disallowed count-based shared feature"
        )
    if len(explanations) != len(test) or len(eligibility_by_id) != len(test):
        raise KMedoidsGenerationError("JSON does not cover every test email exactly once")

    outcome_eligible_counts = {}
    for outcome in ("TP", "TN", "FP", "FN"):
        outcome_ids = frozen_predictions.loc[
            frozen_predictions["outcome_type"].eq(outcome),
            "email_id",
        ].astype(str)
        outcome_eligible_counts[outcome] = int(
            sum(eligibility_by_id[email_id] for email_id in outcome_ids)
        )

    eligible_count = int(sum(eligibility_by_id.values()))
    class_summary = {
        "legitimate": {
            key: value
            for key, value in class_models[0].items()
            if key
            in {
                "train_count",
                "k",
                "silhouette_score",
                "candidate_scores",
                "medoid_ids",
                "distance_threshold",
            }
        },
        "phishing": {
            key: value
            for key, value in class_models[1].items()
            if key
            in {
                "train_count",
                "k",
                "silhouette_score",
                "candidate_scores",
                "medoid_ids",
                "distance_threshold",
            }
        },
    }
    summary = {
        "status": "PASS",
        "test_total": len(test),
        "eligible": eligible_count,
        "ineligible": len(test) - eligible_count,
        "ineligible_reasons": {
            "distance_above_train_95th_percentile": ineligible_distance,
            "fewer_than_2_shared_features": ineligible_shared_features,
            "missing_representative_content": ineligible_missing_content,
        },
        "zero_vs_zero_violation_count": zero_zero_violations,
        "count_feature_rule_violation_count": count_feature_rule_violations,
        "classes": class_summary,
        "medoid_total": sum(model["k"] for model in class_models.values()),
        "eligible_by_outcome_type": outcome_eligible_counts,
        "validations": {
            "feature_order_matches": True,
            "scaler_fit_on_train_only": True,
            "xgboost_predictions_use_unscaled_features": True,
            "train_test_ids_disjoint": True,
            "frozen_test_predictions_match": True,
            "all_medoids_are_train_emails": True,
            "representative_prediction_classes_match": True,
            "nearest_same_class_medoids_used": True,
            "class_thresholds_use_train_95th_percentile": True,
            "eligible_shared_features_follow_fixed_rule": True,
            "eligible_count_features_follow_positive_nearby_rule": True,
            "ratio_features_use_tree_shap_formatting": True,
            "participant_json_has_no_forbidden_keys": True,
        },
        "maximum_probability_difference": maximum_probability_difference,
    }
    return document, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic participant-facing k-medoids explanations"
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
        document, summary = generate_kmedoids_explanations(
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
        KMedoidsGenerationError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"k-medoids generation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
