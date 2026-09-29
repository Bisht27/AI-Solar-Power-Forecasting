"""
Reproducible training script for the DEPLOYMENT model
(model/best_random_forest.pkl).

Run:
    python scripts/train_model.py --input data/Plant_1_Merged.csv

What this script does:
  1. Loads the merged CSV (generation + weather sensor data), filters to a
     single inverter (SOURCE_KEY_x == "1BY6WEcLGh8j5v7" by default, matching
     the research notebook's choice) and sorts chronologically.
  2. Builds the deployment feature schema in the EXACT required order:
         IRRADIATION, Hour, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE,
         AC_POWER_LAG1
     where AC_POWER_LAG1 = AC_POWER.shift(1) (t-1, strictly past information).
  3. Builds the target: AC_POWER.shift(-1) (t+1, the next 15-minute
     interval) -- same target definition as the research notebook.
  4. TRAIN/INFERENCE CONSISTENCY FIX: the deployment app (utils/feature_
     engineering.py) cannot supply a *measured* MODULE_TEMPERATURE at
     inference time -- Open-Meteo does not report it -- so it always uses
     the heuristic estimate
         Ambient + (Shortwave Radiation / 800) * 20.
     Training this model on the dataset's MEASURED module temperature would
     silently train on a different feature distribution than the one it
     will ever see in production (see docs/VERIFICATION_REPORT.md, Part 7).
     This script therefore trains on the SAME estimator, fed with the
     dataset's own IRRADIATION converted back to W/m^2
     (utils.feature_engineering.irradiation_to_shortwave_radiation), so the
     "MODULE_TEMPERATURE" column the model learns from is computed by
     identically the same function/constants the Streamlit app calls at
     inference. The measured sensor MODULE_TEMPERATURE is intentionally NOT
     used as a training feature; it is only used for reporting the
     estimator's deviation from ground truth in the printed diagnostics.
  5. Validates with sklearn.model_selection.TimeSeriesSplit(n_splits=5),
     reporting MAE / RMSE / R2 per fold and in aggregate -- a strict
     chronological validation; no future observation ever appears in a
     training fold.
  6. Refits a final RandomForestRegressor(n_estimators=200, max_depth=12,
     random_state=42, n_jobs=-1) -- the hyperparameters already present in
     the previously-committed model/best_random_forest.pkl -- on ALL
     available data. No hyperparameter tuning is performed here (see
     docs/VERIFICATION_REPORT.md, Part 10); the TimeSeriesSplit folds above,
     not this final refit, are what should be cited for expected
     out-of-sample performance.
  7. Saves the final model to model/best_random_forest.pkl via joblib,
     reloads it, and prints the model type, n_features_in_,
     feature_names_in_, and get_params() so the artifact can be verified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.feature_engineering import (  # noqa: E402
    estimate_module_temperature,
    irradiation_to_shortwave_radiation,
)

FEATURE_COLUMNS = [
    "IRRADIATION",
    "Hour",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "AC_POWER_LAG1",
]
TARGET_COLUMN = "Target"
DEFAULT_INVERTER = "1BY6WEcLGh8j5v7"
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "best_random_forest.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)


def load_and_prepare(csv_path: str, inverter: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_raw_columns = {
        "DATE_TIME", "SOURCE_KEY_x", "AC_POWER", "IRRADIATION",
        "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE",
    }
    missing = required_raw_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}. "
            "This script expects the same schema as Plant_1_Merged.csv "
            "(Plant_1_Generation_Data.csv left-joined with "
            "Plant_1_Weather_Sensor_Data.csv on DATE_TIME, PLANT_ID)."
        )

    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df[df["SOURCE_KEY_x"] == inverter].copy()
    if df.empty:
        raise ValueError(f"No rows found for inverter '{inverter}'.")

    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    df["Hour"] = df["DATE_TIME"].dt.hour

    # --- Train/inference consistency fix (see module docstring, step 4) ---
    # Measured MODULE_TEMPERATURE is kept only for a diagnostic comparison;
    # the FEATURE actually trained on is the same heuristic estimate the
    # deployed app computes, fed with this row's own irradiation converted
    # back to a W/m^2 value.
    df["MODULE_TEMPERATURE_MEASURED"] = df["MODULE_TEMPERATURE"]
    shortwave_radiation = irradiation_to_shortwave_radiation(df["IRRADIATION"].to_numpy())
    df["MODULE_TEMPERATURE"] = estimate_module_temperature(
        df["AMBIENT_TEMPERATURE"].to_numpy(), shortwave_radiation
    )

    # AC_POWER_LAG1: strictly the previous row's AC_POWER (t-1). Because the
    # frame is sorted chronologically and filtered to one inverter, shift(1)
    # cannot pull in another inverter's or a future reading's value.
    df["AC_POWER_LAG1"] = df["AC_POWER"].shift(1)

    # Target: next interval's AC_POWER (t+1), matching the research notebook.
    df[TARGET_COLUMN] = df["AC_POWER"].shift(-1)

    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    dropped = before - len(df)
    print(f"Dropped {dropped} row(s) with NaNs introduced by lag/target shifting "
          f"(expected: exactly 2, the first and last row of the sorted series, "
          f"unless the source data itself contains gaps).")

    err = df["MODULE_TEMPERATURE"] - df["MODULE_TEMPERATURE_MEASURED"]
    print(
        "Module-temperature estimator vs. measured sensor value (diagnostic "
        f"only, not used for training): bias {err.mean():+.2f} C, "
        f"MAE {err.abs().mean():.2f} C, RMSE {np.sqrt((err**2).mean()):.2f} C "
        f"over {len(df)} rows."
    )

    return df


def run_timeseries_validation(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_idx, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = r2_score(y_test, predictions)

        fold_results.append(
            {
                "fold": fold_idx,
                "train_samples": int(len(train_index)),
                "test_samples": int(len(test_index)),
                "mae": float(mae),
                "rmse": rmse,
                "r2": float(r2),
            }
        )
        print(
            f"Fold {fold_idx}: train={len(train_index)} test={len(test_index)} "
            f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f}"
        )
        # MAPE is intentionally NOT reported: this dataset contains
        # zero/night-time AC_POWER values, which make conventional MAPE
        # numerically unstable (division by ~0).

    agg = {
        "mae": float(np.mean([f["mae"] for f in fold_results])),
        "rmse": float(np.mean([f["rmse"] for f in fold_results])),
        "r2": float(np.mean([f["r2"] for f in fold_results])),
    }
    print("\nAggregate (mean across folds):")
    print(f"  MAE  : {agg['mae']:.3f}")
    print(f"  RMSE : {agg['rmse']:.3f}")
    print(f"  R2   : {agg['r2']:.4f}")

    return {"folds": fold_results, "aggregate": agg}


def train_final_model(df: pd.DataFrame) -> RandomForestRegressor:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X, y)
    return model


def save_and_verify(model: RandomForestRegressor, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"\nSaved model to {os.path.abspath(output_path)}")

    reloaded = joblib.load(output_path)
    print("\nVerification (reloaded from disk):")
    print("  Model type        :", type(reloaded))
    print("  n_features_in_    :", getattr(reloaded, "n_features_in_", "N/A"))
    print("  feature_names_in_ :", list(getattr(reloaded, "feature_names_in_", [])))
    print("  Parameters        :", reloaded.get_params())

    expected = FEATURE_COLUMNS
    actual = list(getattr(reloaded, "feature_names_in_", []))
    if actual != expected:
        print(
            f"\n[WARNING] feature_names_in_ {actual} does not match the "
            f"expected deployment schema {expected}. Do not deploy this "
            "artifact until this is resolved."
        )
    else:
        print("\n[OK] feature_names_in_ matches the expected deployment schema.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=os.path.join("data", "Plant_1_Merged.csv"),
        help="Path to the merged solar dataset CSV.",
    )
    parser.add_argument("--inverter", default=DEFAULT_INVERTER)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output", default=MODEL_OUTPUT_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("=" * 70)
        print("[DATA REQUIRED]")
        print(f"Input file not found: {args.input}")
        print(
            "The merged solar dataset is not present at this path. This "
            "script cannot fabricate it. Place Plant_1_Merged.csv there (or "
            "pass --input) and re-run."
        )
        print("=" * 70)
        sys.exit(1)

    df = load_and_prepare(args.input, args.inverter)
    print(f"\nPrepared {len(df)} chronologically-sorted samples for inverter "
          f"'{args.inverter}'.\n")

    print("=" * 70)
    print("TIME-SERIES CROSS-VALIDATION (chronological, no shuffling)")
    print("=" * 70)
    validation = run_timeseries_validation(df, n_splits=args.n_splits)

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL AVAILABLE DATA")
    print("=" * 70)
    final_model = train_final_model(df)
    save_and_verify(final_model, args.output)

    importance = dict(zip(FEATURE_COLUMNS, [float(x) for x in final_model.feature_importances_]))
    print("\nDeployment-model feature importance (final refit, impurity-based):")
    for k, v in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "deployment_training.json"), "w") as f:
        json.dump(
            {
                "inverter": args.inverter,
                "n_samples": len(df),
                "features": FEATURE_COLUMNS,
                "rf_params": RF_PARAMS,
                "validation": validation,
                "feature_importance": importance,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
