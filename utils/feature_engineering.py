"""Converts raw live weather data into the exact feature dataframe the
trained Random Forest model expects.

IMPORTANT: the model was trained with features in this EXACT order:
    1. IRRADIATION
    2. Hour
    3. MODULE_TEMPERATURE
    4. AMBIENT_TEMPERATURE
    5. AC_POWER_LAG1
This order must never change.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]

# ---------------------------------------------------------------------------
# Feature scales -- VERIFIED against Plant_1_Merged.csv (see
# docs/VERIFICATION_REPORT.md, Part 8):
#   * The training column IRRADIATION is on a kW/m^2 scale (observed range for
#     the training inverter: 0.000 .. 1.222; daytime peaks 1.0-1.2).
#   * Open-Meteo `shortwave_radiation` is global horizontal irradiance in W/m^2.
#   => IRRADIATION = shortwave_radiation / 1000 is the correct unit conversion.
#   Caveat that cannot be resolved from the data: the plant's sensor type
#   (plane-of-array vs horizontal) is undocumented, so a residual systematic
#   offset between Open-Meteo GHI and the plant sensor is possible.
# ---------------------------------------------------------------------------
RADIATION_TO_IRRADIATION_DIVISOR = 1000.0  # W/m^2 -> kW/m^2
# Upper clip. The previous value (1.2) sat INSIDE the training range (max 1.222),
# silently truncating legitimate values. A Random Forest cannot extrapolate
# beyond its training range anyway, so 1.3 only changes the displayed value,
# not the model output, for inputs above the training maximum.
IRRADIATION_CLIP_MIN, IRRADIATION_CLIP_MAX = 0.0, 1.3

# ---------------------------------------------------------------------------
# MODULE_TEMPERATURE: in the training data this is a MEASURED value from the
# plant's weather sensor. Open-Meteo provides no module temperature, so at
# inference it must be ESTIMATED. To keep training and inference on the same
# distribution the model is trained on the SAME estimate (this function is
# used by scripts/train_model.py), not on the measured sensor value.
# ---------------------------------------------------------------------------
MODULE_TEMP_REFERENCE_RADIATION = 800.0   # W/m^2
MODULE_TEMP_RISE_AT_REFERENCE = 20.0      # deg C
MODULE_TEMP_CLIP_MIN, MODULE_TEMP_CLIP_MAX = -10.0, 85.0


def shortwave_radiation_to_irradiation(shortwave_radiation):
    """Converts Open-Meteo shortwave radiation (W/m^2) into the model's
    IRRADIATION feature scale (kW/m^2), clipped to [0, 1.3].
    Accepts scalars (returns float) or array-likes (returns ndarray).
    """
    irradiation = np.asarray(shortwave_radiation, dtype=float) / RADIATION_TO_IRRADIATION_DIVISOR
    clipped = np.clip(irradiation, IRRADIATION_CLIP_MIN, IRRADIATION_CLIP_MAX)
    return float(clipped) if clipped.ndim == 0 else clipped


def irradiation_to_shortwave_radiation(irradiation):
    """Inverse of :func:`shortwave_radiation_to_irradiation` (kW/m^2 -> W/m^2).
    Used at TRAINING time to feed the module-temperature estimator with the
    same W/m^2 quantity it receives at inference."""
    return np.asarray(irradiation, dtype=float) * RADIATION_TO_IRRADIATION_DIVISOR


def estimate_module_temperature(ambient_temperature, shortwave_radiation):
    """HEURISTIC estimate of solar module temperature (Open-Meteo has none).

        Module Temperature = Ambient + (Shortwave Radiation / 800) * 20

    This is an engineering approximation, not a physical model. On the
    training inverter's data it has MAE 2.10 C and bias +0.14 C against the
    measured sensor, but under-predicts the module-minus-ambient rise at high
    irradiance (mean error -2.65 C where IRRADIATION > 0.5) and cannot
    reproduce the ~-1.8 C night-time module-below-ambient offset.
    Because it is a deterministic function of the other two inputs it adds no
    independent information to the Random Forest.

    Works on scalars (returns float) or array-likes (returns ndarray).
    """
    estimated = (
        np.asarray(ambient_temperature, dtype=float)
        + (np.asarray(shortwave_radiation, dtype=float) / MODULE_TEMP_REFERENCE_RADIATION)
        * MODULE_TEMP_RISE_AT_REFERENCE
    )
    clipped = np.clip(estimated, MODULE_TEMP_CLIP_MIN, MODULE_TEMP_CLIP_MAX)
    if clipped.ndim == 0:
        if float(clipped) != float(estimated):
            logger.info("Module temperature %.2f clipped to %.2f", float(estimated), float(clipped))
        return float(clipped)
    return clipped


def hour_from_timestamp(timestamp: str | None) -> int | None:
    """Extracts the hour-of-day (0-23) from an Open-Meteo local timestamp such
    as ``2026-09-24T14:15``. Open-Meteo is queried with ``timezone=auto`` so
    this is the hour at the *selected location*, which is what the model's
    ``Hour`` feature was trained on (plant-local time). Returns None if the
    timestamp is missing/unparseable."""
    if not timestamp:
        return None
    try:
        return int(datetime.fromisoformat(str(timestamp)).hour)
    except (ValueError, TypeError):
        return None


def build_feature_dataframe(
    irradiation: float,
    hour: int,
    module_temperature: float,
    ambient_temperature: float,
    ac_power_lag1: float = 0.0,
) -> pd.DataFrame:
    """Builds the single-row dataframe in the exact column order the model expects.

    ``ac_power_lag1`` is the AC power predicted (or observed) for the
    previous 15-minute interval. Pass 0.0 when there is no prior reading yet
    (e.g. the very first prediction of a session).
    """
    return pd.DataFrame(
        [[irradiation, hour, module_temperature, ambient_temperature, ac_power_lag1]],
        columns=FEATURE_COLUMNS,
    )


def engineer_features_from_weather(
    ambient_temperature: float,
    shortwave_radiation: float,
    hour: int,
    ac_power_lag1: float = 0.0,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end: raw weather -> engineered features -> model-ready dataframe.

    Args:
        ac_power_lag1: The previous interval's AC power output (kW). This
            captures short-term momentum (e.g. a passing cloud) that the
            instantaneous weather features alone can't express. Defaults to
            0.0 when no prior reading is available yet.

    Returns the feature dataframe plus a dict of the intermediate values
    (irradiation, module_temperature) for display in the UI.
    """
    irradiation = shortwave_radiation_to_irradiation(shortwave_radiation)
    module_temperature = estimate_module_temperature(ambient_temperature, shortwave_radiation)

    features_df = build_feature_dataframe(
        irradiation=irradiation,
        hour=hour,
        module_temperature=module_temperature,
        ambient_temperature=ambient_temperature,
        ac_power_lag1=ac_power_lag1,
    )

    intermediate = {
        "irradiation": irradiation,
        "module_temperature": module_temperature,
        "hour": hour,
        "ac_power_lag1": ac_power_lag1,
    }
    return features_df, intermediate


def _daylight_factor(hour: int) -> float:
    """Simple bell-shaped daylight curve peaking at midday, zero at night."""
    import math

    if hour < 6 or hour > 18:
        return 0.0
    return max(0.0, math.sin((hour - 6) / 12 * math.pi))


def simulate_intraday_curve(
    model,
    ambient_temperature: float,
    shortwave_radiation: float,
    current_hour: int,
) -> pd.DataFrame:
    """Simulates a full 0-23h AC power curve for "today" using the current
    weather reading as an anchor point.

    Rather than requiring historical data, this scales the *current* solar
    radiation up/down across the day using a daylight bell curve normalized
    so that the simulated value at ``current_hour`` matches today's actual
    reading. AC_POWER_LAG1 is chained hour-to-hour (each hour's prediction
    feeds the next hour's lag feature), giving a smooth, self-consistent
    intraday trend for the dashboard.

    This is a visualization aid, not a certified forecast for other hours.
    """
    anchor_factor = _daylight_factor(current_hour)
    # Avoid divide-by-zero at night; fall back to a nominal peak-hour scale.
    peak_radiation = shortwave_radiation / anchor_factor if anchor_factor > 1e-3 else shortwave_radiation

    rows = []
    ac_power_lag1 = 0.0
    for hour in range(24):
        factor = _daylight_factor(hour)
        hour_radiation = max(0.0, peak_radiation * factor)
        hour_irradiation = shortwave_radiation_to_irradiation(hour_radiation)
        hour_module_temp = estimate_module_temperature(ambient_temperature, hour_radiation)

        features_df = build_feature_dataframe(
            irradiation=hour_irradiation,
            hour=hour,
            module_temperature=hour_module_temp,
            ambient_temperature=ambient_temperature,
            ac_power_lag1=ac_power_lag1,
        )
        predicted = float(max(0.0, model.predict(features_df)[0]))
        rows.append({"hour": hour, "predicted_ac_power": predicted})
        ac_power_lag1 = predicted

    return pd.DataFrame(rows)
