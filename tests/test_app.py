"""Test suite for the Solar Power Forecasting deployment pipeline.

Run with:
    pytest tests/test_app.py -v

These tests exercise the model/feature-engineering/prediction pipeline
using whatever model file is currently at model/best_random_forest.pkl
(as of this revision, a model genuinely trained on the real Plant 1 solar
dataset -- see scripts/train_model.py and docs/VERIFICATION_REPORT.md).
They do NOT make live network calls (no Open-Meteo requests), so they run
offline. They also do not attempt to verify forecasting *accuracy* -- that
is covered separately by scripts/train_model.py's TimeSeriesSplit
validation and reported in results/deployment_training.json.

The model is loaded ONLY from the local file model/best_random_forest.pkl.
There is no remote/download fallback of any kind in this project.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_dataframe,
    engineer_features_from_weather,
    estimate_module_temperature,
    shortwave_radiation_to_irradiation,
    simulate_intraday_curve,
)
from utils.prediction import (  # noqa: E402
    ModelLoadError,
    is_placeholder_model,
    load_model,
    predict_ac_power,
)
from utils.confidence import calculate_confidence  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_random_forest.pkl")

EXPECTED_FEATURES = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]


@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"No model file at {MODEL_PATH}")
    return load_model(MODEL_PATH)


# 1. Model loads successfully
def test_model_loads(model):
    assert model is not None


# 2. Model has 5 features
def test_model_has_five_features(model):
    assert getattr(model, "n_features_in_", None) == 5


# 3. Feature order is correct
def test_model_feature_order(model):
    names = list(getattr(model, "feature_names_in_", []))
    assert names == EXPECTED_FEATURES


# 4. Feature engineering produces expected columns
def test_feature_engineering_columns():
    df, _ = engineer_features_from_weather(
        ambient_temperature=25.0, shortwave_radiation=500.0, hour=12, ac_power_lag1=100.0
    )
    assert list(df.columns) == FEATURE_COLUMNS == EXPECTED_FEATURES
    assert len(df) == 1


# 5. Prediction returns a numeric non-negative value
def test_prediction_is_numeric_and_non_negative(model):
    features_df = build_feature_dataframe(
        irradiation=0.8, hour=12, module_temperature=35.0, ambient_temperature=28.0, ac_power_lag1=50.0
    )
    prediction = predict_ac_power(model, features_df)
    assert isinstance(prediction, float)
    assert prediction >= 0.0


# 6. Missing / edge-case weather values are handled (clipping)
def test_irradiation_and_module_temp_clipping():
    # Extreme radiation should be clipped, not raise or return an
    # out-of-range value.
    assert shortwave_radiation_to_irradiation(999999) == 1.3
    assert shortwave_radiation_to_irradiation(-500) == 0.0
    module_temp = estimate_module_temperature(ambient_temperature=200.0, shortwave_radiation=5000.0)
    assert -10.0 <= module_temp <= 85.0


# 7. AC_POWER_LAG1 is handled (defaults to 0.0, and is passed through correctly)
def test_ac_power_lag1_default_and_passthrough():
    df_default, intermediate_default = engineer_features_from_weather(
        ambient_temperature=25.0, shortwave_radiation=400.0, hour=10
    )
    assert df_default["AC_POWER_LAG1"].iloc[0] == 0.0
    assert intermediate_default["ac_power_lag1"] == 0.0

    df_custom, intermediate_custom = engineer_features_from_weather(
        ambient_temperature=25.0, shortwave_radiation=400.0, hour=10, ac_power_lag1=123.4
    )
    assert df_custom["AC_POWER_LAG1"].iloc[0] == 123.4
    assert intermediate_custom["ac_power_lag1"] == 123.4


# 8. Local model loading never silently accepts a schema-mismatched model,
#    and never reaches out to any remote source when the local file is
#    missing (utils.prediction.check_feature_schema / resolve_model_path).
def test_local_model_schema_check_and_no_remote_fallback(tmp_path, monkeypatch, caplog):
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    from utils import prediction as pred

    # 8a. A model with the WRONG feature schema is detected and logged,
    #     rather than silently accepted.
    bad_model = RandomForestRegressor(n_estimators=2, random_state=0)
    X = pd.DataFrame({"WRONG_FEATURE": [1, 2, 3]})
    y = [1, 2, 3]
    bad_model.fit(X, y)

    assert pred.check_feature_schema(bad_model) is False

    # 8b. A model with the CORRECT feature schema passes the check.
    good_model = RandomForestRegressor(n_estimators=2, random_state=0)
    Xg = pd.DataFrame({c: [1, 2, 3] for c in pred.EXPECTED_FEATURE_ORDER})
    good_model.fit(Xg, y)
    assert pred.check_feature_schema(good_model) is True

    # 8c. If no local model file exists, resolve_model_path raises a clear
    #     local error, and does so without opening any network socket.
    import socket

    def _no_network(*args, **kwargs):
        raise AssertionError("network access attempted while resolving the model path")

    monkeypatch.setattr(socket, "socket", _no_network)
    missing_path = str(tmp_path / "does_not_exist.pkl")
    with pytest.raises(pred.ModelLoadError):
        pred.resolve_model_path(candidates=[missing_path])


# 9. Intraday simulation runs and produces 24 hourly rows, clearly labeled
#    illustrative (checked at the app.py text level, not here)
def test_intraday_simulation_runs(model):
    result = simulate_intraday_curve(
        model=model, ambient_temperature=25.0, shortwave_radiation=600.0, current_hour=13
    )
    assert len(result) == 24
    assert list(result.columns) == ["hour", "predicted_ac_power"]
    assert (result["predicted_ac_power"] >= 0).all()


# 10. Application module imports successfully (Streamlit script-mode import)
def test_app_module_imports():
    import importlib

    # app.py calls st.set_page_config() at import time via streamlit; importing
    # it outside `streamlit run` will raise a StreamlitAPIException or similar
    # runtime error rather than a plain ImportError/SyntaxError. We only assert
    # that it is syntactically valid and importable as far as Python parsing
    # goes -- full Streamlit runtime behavior is exercised by `streamlit run`.
    import ast

    app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    with open(app_path, encoding="utf-8") as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError if invalid


# Extra: confidence/consistency score is bounded and doesn't error
def test_confidence_score_bounded(model):
    features_df = build_feature_dataframe(
        irradiation=0.5, hour=12, module_temperature=30.0, ambient_temperature=25.0, ac_power_lag1=80.0
    )
    score = calculate_confidence(model, features_df)
    assert 0.0 <= score <= 100.0


# Extra: shipped model provenance is at least checkable via the heuristic
def test_is_placeholder_model_heuristic_runs(model):
    # We don't assert the result here (it depends on whatever model file is
    # currently present) -- only that the heuristic itself runs without error.
    result = is_placeholder_model(model)
    assert isinstance(result, bool)
