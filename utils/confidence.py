"""Computes a tree-disagreement-based model-consistency score by measuring
how much the individual trees in the Random Forest disagree with each other
on a given input.

IMPORTANT: this is NOT a calibrated probability, NOT a statistical
confidence interval, and NOT a validated accuracy estimate. It has not been
calibrated against held-out data. A tight spread across trees means the
trees agree with each other on this particular input; it says nothing
about whether that agreed-upon value is actually close to the true AC
power. Report/display it as "model-consistency score" or
"tree-disagreement-based confidence estimate", never as "confidence
interval", "probability", or "% accuracy".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

# The AC power range (in the same units as the model's target) used to
# normalize tree-disagreement into a 0-100% confidence score. Set this to
# your plant's realistic max AC power for a well-calibrated confidence value.
DEFAULT_MAX_PREDICTION_RANGE = 1000.0


def calculate_confidence(
    model: Any,
    features_df: pd.DataFrame,
    max_prediction_range: float = DEFAULT_MAX_PREDICTION_RANGE,
) -> float:
    """Returns a tree-disagreement-based model-consistency score (0-100) for
    the current prediction. See module docstring: this is NOT a calibrated
    probability or confidence interval.

    Formula: 100 - (std_dev(tree_predictions) / max_prediction_range) * 100,
    clipped to [0, 100].

    If the model does not expose individual trees (i.e. it isn't a bagged
    ensemble like RandomForestRegressor), a neutral default of 75.0 is
    returned rather than failing the whole prediction flow. This default is
    an arbitrary placeholder, not a measured value, and should be labeled
    as "unavailable" or similarly caveated in any UI that surfaces it.
    """
    estimators = getattr(model, "estimators_", None)
    if not estimators:
        logger.info("Model has no .estimators_; returning default consistency score.")
        return 75.0

    try:
        # Individual trees inside a fitted RandomForestRegressor are each
        # trained on plain NumPy arrays internally (sklearn does not thread
        # pandas column-name metadata down to the sub-estimators), so calling
        # tree.predict() with a DataFrame triggers a benign but noisy
        # "X has feature names, but ... was fitted without feature names"
        # UserWarning on every single tree. Passing the underlying NumPy
        # values instead avoids the warning without changing the prediction
        # in any way -- the forest-level model.predict() elsewhere in this
        # codebase is unaffected and still receives the DataFrame.
        feature_values = features_df.to_numpy()
        tree_predictions = np.array([tree.predict(feature_values)[0] for tree in estimators])
    except Exception:  # noqa: BLE001
        logger.exception(
            "Failed to gather per-tree predictions; returning default consistency score."
        )
        return 75.0

    std_dev = float(np.std(tree_predictions))
    safe_range = max(max_prediction_range, 1e-6)  # avoid division by zero
    consistency_score = 100.0 - (std_dev / safe_range) * 100.0
    return float(min(max(consistency_score, 0.0), 100.0))


# Backward-compatible alias with a name that doesn't overstate what the
# metric is. Prefer this name in new code.
calculate_model_consistency_score = calculate_confidence
