"""Loads the pre-trained Random Forest model and runs predictions.

This module NEVER trains or retrains a model - it only loads an existing
local artifact and calls .predict() on it, per the project requirements.

The model is loaded ONLY from a local file (model/best_random_forest.pkl
by default). There is no remote/network fallback of any kind: if the local
file is missing, load_model() raises ModelLoadError with a clear message
rather than attempting to fetch a model from anywhere else. An earlier
revision of this project fell back to downloading a model from Google
Drive when no local file was present; that fallback has been removed
entirely (its provenance could never be verified from this project's
environment, and a purely local, explicit failure is safer than silently
substituting an unverified remote artifact).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import joblib
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATHS = [
    os.path.join("model", "best_random_forest.pkl"),
    os.path.join("model", "rf_model.pkl"),
]

# Set to "1" to deliberately allow the app to serve the synthetic demo
# placeholder model (UI smoke-testing only). Off by default: without it,
# load_model() refuses a placeholder instead of silently serving it.
ALLOW_DEMO_MODEL_ENV = "SOLAR_ALLOW_DEMO_MODEL"

EXPECTED_FEATURE_ORDER = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]


class ModelLoadError(Exception):
    """Raised when the trained model file cannot be found or loaded."""


class PredictionError(Exception):
    """Raised when the model fails to produce a prediction."""


def resolve_model_path(candidates: list[str] | None = None) -> str:
    """Finds the first existing local model file.

    Raises:
        ModelLoadError: if none of the candidate paths exist. This function
        never downloads or otherwise fetches a model from a remote source.
    """

    for path in candidates or DEFAULT_MODEL_PATHS:
        if os.path.exists(path):
            logger.info("Found model: %s", path)
            return path

    raise ModelLoadError(
        "No local model file found. Expected one of: "
        f"{candidates or DEFAULT_MODEL_PATHS}. Train one with "
        "`python scripts/train_model.py --input data/Plant_1_Merged.csv`, "
        "then place the resulting model/best_random_forest.pkl alongside "
        "the app. This project does not download a model from any remote "
        "source."
    )


def check_feature_schema(model: Any, expected: list[str] | None = None) -> bool:
    """Best-effort sanity check: does ``model.feature_names_in_`` match the
    deployment schema this app expects, in the exact same order?

    Returns True if the schema matches (or if the model doesn't expose
    feature_names_in_ at all, e.g. an older sklearn version -- there is
    nothing to compare against in that case, so this is not treated as a
    mismatch). Returns False, and logs a warning, if a mismatch is found.
    This does not verify whether the model was trained on real vs.
    synthetic data -- only that its declared input schema is consistent
    with what utils/feature_engineering.py constructs at inference time.
    """
    expected = expected or EXPECTED_FEATURE_ORDER
    actual = list(getattr(model, "feature_names_in_", []))
    if actual and actual != expected:
        logger.warning(
            "Loaded model's feature_names_in_ %s does not match the expected "
            "schema %s. Predictions will likely fail or be meaningless.",
            actual, expected,
        )
        return False
    return True


def metadata_path_for(model_path: str) -> str:
    """Path of the provenance sidecar written by scripts/train_model.py."""
    return os.path.splitext(model_path)[0] + ".meta.json"


def load_model_metadata(model_path: str) -> dict | None:
    """Returns the provenance sidecar for ``model_path``, or None if it is
    missing/unreadable. Never raises."""
    try:
        with open(metadata_path_for(model_path), "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def verify_model_provenance(model_path: str, metadata: dict | None = None) -> bool:
    """True only if a sidecar exists, declares real-data training, and its
    recorded SHA-256 matches the model file currently on disk (i.e. the
    file has not been swapped or overwritten since scripts/train_model.py
    produced it)."""
    metadata = metadata if metadata is not None else load_model_metadata(model_path)
    if not metadata or metadata.get("trained_on") != "real_data":
        return False
    digest = hashlib.sha256()
    with open(model_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == metadata.get("model_sha256")


def _demo_model_allowed() -> bool:
    return os.environ.get(ALLOW_DEMO_MODEL_ENV, "").strip() == "1"


def load_model(model_path: str | None = None, allow_placeholder: bool | None = None) -> Any:
    """Loads the trained Random Forest model from disk via joblib.

    The synthetic demo placeholder is refused unless ``allow_placeholder``
    is True (or the SOLAR_ALLOW_DEMO_MODEL=1 environment variable is set),
    so the app can never silently serve it.

    Raises:
        ModelLoadError: if the file is missing, fails to deserialize, or is
        detected as the synthetic placeholder and demo use was not allowed.
    """
    path = model_path or resolve_model_path()
    try:
        model = joblib.load(path)
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model file not found at '{path}'.") from exc
    except Exception as exc:  # noqa: BLE001 - surface any deserialization issue clearly
        logger.exception("Failed to load model from %s", path)
        raise ModelLoadError(f"Could not load the model file at '{path}': {exc}") from exc

    logger.info("Loaded model from %s (%s)", path, type(model).__name__)
    check_feature_schema(model)

    metadata = load_model_metadata(path)
    if verify_model_provenance(path, metadata):
        logger.info(
            "Model provenance verified: trained on real data (%s, %s samples) at %s.",
            metadata.get("dataset_file"), metadata.get("n_samples"), metadata.get("trained_at_utc"),
        )
        recorded = metadata.get("sklearn_version")
        if recorded:
            import sklearn

            if recorded != sklearn.__version__:
                logger.warning(
                    "Model was trained with scikit-learn %s but %s is installed; "
                    "unpickling across versions can give wrong results.",
                    recorded, sklearn.__version__,
                )
    else:
        if metadata is not None:
            logger.warning(
                "Provenance sidecar for %s does not match the model file; "
                "the file changed after training. Falling back to the placeholder heuristic.", path,
            )
        if is_placeholder_model(model):
            if allow_placeholder if allow_placeholder is not None else _demo_model_allowed():
                logger.warning("Serving the SYNTHETIC DEMO placeholder model (demo use explicitly allowed).")
            else:
                raise ModelLoadError(
                    f"'{path}' is the synthetic demo placeholder model, not a model trained on real "
                    "solar data, so it was not loaded. Train the real model with "
                    "`python scripts/train_model.py --input data/Plant_1_Merged.csv`. "
                    f"(For a UI-only smoke test you can set {ALLOW_DEMO_MODEL_ENV}=1.)"
                )
    return model



def describe_model(model_path: str | None = None) -> dict:
    """Provenance summary for the UI: ``{"verified": bool, "metadata": dict|None}``.
    Never raises."""
    try:
        path = model_path or resolve_model_path()
        metadata = load_model_metadata(path)
        return {"verified": verify_model_provenance(path, metadata), "metadata": metadata}
    except Exception:  # noqa: BLE001
        return {"verified": False, "metadata": None}


def is_placeholder_model(model: Any) -> bool:
    """Best-effort heuristic check for whether ``model`` is the synthetic
    demo placeholder produced by scripts/generate_demo_placeholder_model.py,
    rather than a model trained on real solar data.

    This is NOT a cryptographic or guaranteed check -- it compares
    hyperparameters and a handful of sample predictions against a freshly
    regenerated placeholder (same fixed seed). A genuinely-trained model
    that happens to use identical hyperparameters and produces coincidentally
    similar predictions on these specific inputs could, in principle, be
    misclassified. Treat a ``True`` result as "very likely the placeholder"
    and a ``False`` result as "not confirmed to be the placeholder" rather
    than a hard guarantee either way.
    """
    try:
        params = model.get_params()
    except Exception:  # noqa: BLE001
        return False

    placeholder_params = {"n_estimators": 200, "max_depth": 12, "random_state": 42}
    if not all(params.get(k) == v for k, v in placeholder_params.items()):
        return False

    expected_features = ["IRRADIATION", "Hour", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE", "AC_POWER_LAG1"]
    actual_features = list(getattr(model, "feature_names_in_", []))
    if actual_features != expected_features:
        return False

    try:
        from scripts.generate_demo_placeholder_model import make_synthetic_dataset

        features, target = make_synthetic_dataset()
        reference = joblib_fit_reference(features, target)
        sample = features.iloc[:5]
        diff = (reference.predict(sample) - model.predict(sample))
        return bool((abs(diff) < 5.0).all())
    except Exception:  # noqa: BLE001
        logger.info("Could not run placeholder-detection heuristic.", exc_info=True)
        return False


def joblib_fit_reference(features: pd.DataFrame, target) -> Any:
    """Fits a fresh reference RandomForestRegressor with the placeholder's
    exact hyperparameters, used only by is_placeholder_model()'s heuristic
    comparison."""
    from sklearn.ensemble import RandomForestRegressor

    reference = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    reference.fit(features, target)
    return reference


def predict_ac_power(model: Any, features_df: pd.DataFrame) -> float:
    """Runs the model's prediction and returns a single float (kW).

    Raises:
        PredictionError: if the model fails to produce a prediction, e.g. due
        to a feature mismatch.
    """
    try:
        prediction = model.predict(features_df)
    except Exception as exc:  # noqa: BLE001 - convert any sklearn error into a friendly one
        logger.exception("Prediction failed")
        raise PredictionError(
            "The model could not generate a prediction from the current features. "
            "This usually means the loaded model expects different input features."
        ) from exc

    value = float(prediction[0])
    return max(0.0, value)  # AC power cannot be negative
