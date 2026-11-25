"""
============================================================================
 DEVELOPMENT-ONLY DEMO UTILITY -- DO NOT USE FOR RESEARCH OR PRODUCTION
============================================================================

This script fits a RandomForestRegressor on 100% SYNTHETIC, RANDOMLY
GENERATED data. It exists ONLY so a developer can see the Streamlit
dashboard render end-to-end (gauges, charts, model-info panel) before a
real, properly trained model is available.

DO NOT:
  - Use the model this script produces as a research result.
  - Report metrics computed against this model as solar forecasting
    performance.
  - Deploy this model to any environment a real user will see predictions
    from.
  - Confuse its output with the real model. It is written to
    model/demo_placeholder_model.pkl (never to best_random_forest.pkl), and
    the app refuses to load a placeholder unless SOLAR_ALLOW_DEMO_MODEL=1
    is set explicitly.

For a genuine model, use `scripts/train_model.py`, which trains on the real
historical solar dataset (Plant_1_Merged.csv) using a chronological
(TimeSeriesSplit) validation methodology. See README.md ("Research vs.
Deployment Model") for details.

This file is never imported or invoked by app.py. Run it manually only if
you understand the above and specifically want a throwaway demo model:

    python scripts/generate_demo_placeholder_model.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Deliberately NOT model/best_random_forest.pkl: this script must never be able
# to overwrite the real, genuinely-trained model.
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "demo_placeholder_model.pkl")
FEATURE_COLUMNS = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]


def make_synthetic_dataset(n_samples: int = 4000, seed: int = 42):
    """Generates purely synthetic, randomly-sampled data. This is NOT real
    solar telemetry and must never be described as such."""
    rng = np.random.default_rng(seed)

    irradiation = rng.uniform(0, 1.1, n_samples)
    hour = rng.integers(0, 24, n_samples)
    ambient_temperature = rng.uniform(15, 42, n_samples)
    module_temperature = ambient_temperature + irradiation * 25 + rng.normal(0, 1.5, n_samples)

    daylight_factor = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, 1)
    base_power = irradiation * daylight_factor * 900
    temp_derate = 1 - np.clip((module_temperature - 25) * 0.004, 0, 0.25)
    noise = rng.normal(0, 15, n_samples)

    ac_power = np.clip(base_power * temp_derate + noise, 0, None)

    lag_noise = rng.normal(0, 25, n_samples)
    ac_power_lag1 = np.clip(ac_power * rng.uniform(0.85, 1.05, n_samples) + lag_noise, 0, None)

    features = pd.DataFrame(
        np.column_stack(
            [irradiation, hour, module_temperature, ambient_temperature, ac_power_lag1]
        ),
        columns=FEATURE_COLUMNS,
    )
    return features, ac_power


def main() -> None:
    print("=" * 70)
    print("WARNING: This generates a DEMO-ONLY model trained on SYNTHETIC data.")
    print("It is NOT a solar-forecasting model. Do not use it for anything")
    print("beyond confirming the Streamlit UI renders correctly.")
    print("=" * 70)

    if os.path.exists(OUTPUT_PATH):
        reply = input(
            f"\n'{OUTPUT_PATH}' already exists. Overwrite it with a synthetic "
            "placeholder? This will destroy whatever model is currently there. "
            "Type 'yes' to continue: "
        )
        if reply.strip().lower() != "yes":
            print("Aborted. Existing model left untouched.")
            sys.exit(0)

    features, target = make_synthetic_dataset()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, target)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"\nDemo placeholder model written to {os.path.abspath(OUTPUT_PATH)}")
    print("Replace this file with a genuinely trained model (see scripts/train_model.py)")
    print("before using this app for anything beyond a UI demo.")


if __name__ == "__main__":
    main()
