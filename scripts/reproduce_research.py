"""
Clean-process reproduction of the ORIGINAL two-feature research experiment
(eda.ipynb cells 21-26).

Nothing here depends on notebook state. Everything (splitter, fold counter,
score lists) is created inside this process, which is precisely what the
notebook failed to do (see docs/VERIFICATION_REPORT.md, "Notebook forensics").

Experiment as written in the notebook's code cells:
  * one inverter (SOURCE_KEY_x == 1BY6WEcLGh8j5v7), sorted by DATE_TIME
  * drop SOURCE_KEY_x / SOURCE_KEY_y / PLANT_ID
  * Target = AC_POWER.shift(-1); drop rows with NaN (i.e. the last row)
  * features = [IRRADIATION, DAILY_YIELD]
  * TimeSeriesSplit(n_splits=5) -- the notebook never defines `tscv`, so n_splits=5
    is taken from the project brief ("TimeSeriesSplit with 5 folds")
  * RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
  * per-fold MAE / RMSE / R2, then the mean over folds

Usage:
    python scripts/reproduce_research.py --input data/Plant_1_Merged.csv
Writes results/research_reproduction.json and results/research_feature_importance.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

INVERTER = "1BY6WEcLGh8j5v7"
RESEARCH_FEATURES = ["IRRADIATION", "DAILY_YIELD"]
RF_PARAMS = dict(n_estimators=100, random_state=42, n_jobs=-1)
CLAIMED = {"mae": 46.959, "rmse": 98.166, "r2": 0.9092}


def load_inverter_frame(path: str, inverter: str = INVERTER) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df = df[df["SOURCE_KEY_x"] == inverter]
    return df.sort_values("DATE_TIME").reset_index(drop=True)


def run(df: pd.DataFrame, features=RESEARCH_FEATURES, n_splits=5, rf_params=RF_PARAMS,
        n_rows_override=None):
    df = df.drop(columns=["SOURCE_KEY_x", "SOURCE_KEY_y", "PLANT_ID"], errors="ignore").copy()
    df["Target"] = df["AC_POWER"].shift(-1)
    df = df.dropna()                      # notebook cell 23: drops the last row
    X, y = df[list(features)], df["Target"]
    folds = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        m = RandomForestRegressor(**rf_params).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        folds.append(dict(fold=k, train=len(tr), test=len(te),
                          train_end=str(df["DATE_TIME"].iloc[tr[-1]]),
                          test_start=str(df["DATE_TIME"].iloc[te[0]]),
                          test_end=str(df["DATE_TIME"].iloc[te[-1]]),
                          mae=mean_absolute_error(y.iloc[te], p),
                          rmse=float(np.sqrt(mean_squared_error(y.iloc[te], p))),
                          r2=r2_score(y.iloc[te], p)))
    agg = {k: float(np.mean([f[k] for f in folds])) for k in ("mae", "rmse", "r2")}
    return df, folds, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join("data", "Plant_1_Merged.csv"))
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    raw = load_inverter_frame(a.input)
    print(f"Rows for inverter {INVERTER}: {len(raw)}")
    df, folds, agg = run(raw)
    print(f"Samples after Target=AC_POWER.shift(-1) + dropna: {len(df)}")
    print(f"{'Fold':>4} {'Train':>6} {'Test':>5} {'MAE':>9} {'RMSE':>9} {'R2':>8}")
    for f in folds:
        print(f"{f['fold']:>4} {f['train']:>6} {f['test']:>5} {f['mae']:>9.3f} {f['rmse']:>9.3f} {f['r2']:>8.4f}")
    print(f"Mean of folds  MAE={agg['mae']:.3f}  RMSE={agg['rmse']:.3f}  R2={agg['r2']:.4f}")
    print(f"Claimed        MAE={CLAIMED['mae']}  RMSE={CLAIMED['rmse']}  R2={CLAIMED['r2']}")

    # Fresh feature importance from a model trained on the SAME data / SAME features
    # as the evaluation (final refit on all samples), never from `final_model`.
    final = RandomForestRegressor(**RF_PARAMS).fit(df[RESEARCH_FEATURES], df["Target"])
    imp = pd.DataFrame({"Feature": RESEARCH_FEATURES, "Importance": final.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False)
    print("\nResearch-model feature importance (impurity-based, final refit):\n", imp.to_string(index=False))
    imp.to_csv(os.path.join(a.outdir, "research_feature_importance.csv"), index=False)
    json.dump(dict(features=RESEARCH_FEATURES, rf_params=RF_PARAMS, n_samples=len(df),
                   folds=folds, aggregate=agg, claimed=CLAIMED),
              open(os.path.join(a.outdir, "research_reproduction.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
