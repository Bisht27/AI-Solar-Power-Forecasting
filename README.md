# ☀️ AI Solar AC Power Forecasting System

Live-weather-powered Streamlit dashboard that predicts a solar plant's
**next 15-minute AC power output**, plus a research notebook that explored
the same forecasting problem with a simpler two-feature model.

> **Status, after a full audit + real data + real training run (this
> revision):** `model/best_random_forest.pkl` is now a **genuinely trained**
> Random Forest, fit on the real Plant 1 solar dataset with a documented,
> reproducible procedure (`scripts/train_model.py`). The previously
> committed file was a synthetic placeholder fit on randomly generated data
> — that issue is now resolved for the copy of the repository produced by
> this audit. See "Verification Report" below for exactly what was checked
> and how.

**Project links:** dataset <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data>

---

## Overview

Point the app at any location (by city name, coordinates, or your browser's
current location) and it will:

1. Pull live weather data for that location from Open-Meteo (no API key required).
2. Derive the model's required features from that weather data.
3. Run a Random Forest model to predict AC power for the next 15 minutes.
4. Compute a tree-disagreement-based **model-consistency score** (not a
   calibrated confidence interval — see "Model-Consistency Score" below).
5. Present everything on a card-based dashboard with live gauges, a map, and
   a weather summary.

---

## Verification Report

Full details, methodology, and every numeric result are in
[`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md). Summary:

- **Dataset verified from scratch.** `Plant_1_Generation_Data.csv` (68,778
  rows, 22 inverters) left-joined with `Plant_1_Weather_Sensor_Data.csv`
  (3,182 rows) on `(DATE_TIME, PLANT_ID)` reproduces the 68,778-row,
  15-column merged dataset the research notebook calls `backdata.csv` /
  `Plant_1_Merged.csv`, spanning 2020-05-15 to 2020-06-17 at a 15-minute
  cadence. Filtering to inverter `1BY6WEcLGh8j5v7` gives **3,154** rows,
  matching the previously claimed figure.
- **Original research experiment reproduced from a clean process** (not the
  saved notebook state). The originally reported aggregate metrics
  (MAE 46.959, RMSE 98.166, R² 0.9092) **could not be reproduced** and are
  **not internally consistent even with the notebook's own saved per-fold
  output** (see report Part 3). The correct, freshly computed aggregate for
  the documented two-feature (`IRRADIATION`, `DAILY_YIELD`) experiment is:

  | Metric | Originally claimed | Reproduced (this audit) |
  |---|---|---|
  | MAE  | 46.959 | **65.025** |
  | RMSE | 98.166 | **141.608** |
  | R²   | 0.9092 | **0.8420** |

- **Stale feature-importance chart explained, not repeated.** The
  notebook's `final_model.feature_names_in_` is
  `['IRRADIATION', 'DC_POWER', 'DAILY_YIELD']` — a 3-feature model from an
  earlier, unlabeled run that was never refit after the researcher
  redefined `features = ["IRRADIATION", "DAILY_YIELD"]` for backtesting.
  The `DC_POWER`-dominated importance chart does not describe the evaluated
  2-feature model. A fresh importance chart for the actual research model
  is in `results/figures/research_feature_importance.png`.
- **A genuine 5-feature deployment model has been trained** on the same
  real dataset, chronologically validated with `TimeSeriesSplit(n_splits=5)`:

  | Metric | Deployment model (this audit) |
  |---|---|
  | MAE  | **58.641** |
  | RMSE | **125.161** |
  | R²   | **0.8805** |

  (Per-fold breakdown in `results/deployment_training.json` and the
  verification report.)
- **Train/inference consistency fixed.** `MODULE_TEMPERATURE` is *measured*
  in the historical dataset but can only be *estimated* at live-inference
  time (Open-Meteo has no module-temperature field). Training was changed
  to use the identical estimator function
  (`utils.feature_engineering.estimate_module_temperature`) the app calls
  at inference, rather than silently training on the measured sensor value.
- **`IRRADIATION` scale confirmed.** Training-data `IRRADIATION` is on a
  kW/m² scale (observed max 1.222); Open-Meteo `shortwave_radiation` is
  W/m². `IRRADIATION = shortwave_radiation / 1000` is the correct
  conversion; the upper clip was widened from 1.2 to 1.3 kW/m² since 1.2
  sat *inside* the observed training range and was silently truncating
  legitimate values.
- **No data leakage found**: lag/target shifts, chronological sort, and
  `TimeSeriesSplit` were all checked directly against the real data (report
  Part 11).
- **Model loaded locally only.** The application loads the trained model
  from `model/best_random_forest.pkl` with `joblib`. If the file is missing
  it raises a clear `ModelLoadError`; it never downloads a model.
- **12/12 tests pass, 0 warnings** (`pytest tests/test_app.py -v`), including
  a placeholder-detection heuristic test, which now returns `False` for the
  retrained model.

---

## Research Model vs. Deployment Model

These are **two different experiments** with two different feature sets.
They must not be conflated.

### Research experiment (`eda.ipynb`)

- **Dataset:** `Plant_1_Merged.csv`, filtered to inverter `1BY6WEcLGh8j5v7`
  → 3,154 chronologically sorted samples (3,153 after the target shift
  drops the final row).
- **Features:** `IRRADIATION`, `DAILY_YIELD`
- **Target:** `AC_POWER.shift(-1)` — the AC power at the next chronological
  observation. For the selected inverter this is exactly 15 minutes ahead
  in 3,143 of 3,153 consecutive transitions; 11 transitions span an
  irregular gap longer than 15 minutes (real sensor/telemetry gaps in the
  underlying data). "Next 15-minute reading" is therefore an approximation
  that holds for most, not all, observations.
- **Model:** `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`
- **Validation:** `TimeSeriesSplit(n_splits=5)` (chronological, no shuffling)
- **Verified aggregate metrics (this audit, clean re-run):** MAE = 65.025,
  RMSE = 141.608, R² = 0.8420 — **not** the previously claimed
  46.959 / 98.166 / 0.9092 (see Verification Report Part 3 for why those
  numbers don't hold up and could not be reproduced under any feature
  configuration tried).
  - R² = 0.8420 means **≈84.18% of variance is explained** under this
    evaluation — this is not "84.18% accuracy" and should never be
    described that way.
  - MAPE is intentionally **not reported**: the target series has many
    zero/night-time values, which make conventional MAPE numerically
    unstable (division by ~0).

### Deployment model (`app.py` / `model/best_random_forest.pkl`)

- **Features (exact order):** `IRRADIATION`, `Hour`, `MODULE_TEMPERATURE`,
  `AMBIENT_TEMPERATURE`, `AC_POWER_LAG1`
- **Target:** AC power at the next chronological reading (approximately
  15 minutes ahead for most observations; see the research model's target
  note above for the same caveat about irregular gaps in the historical
  training data — live inference always predicts one step ahead of
  whatever cadence Open-Meteo/the app polls at).
- **Model:** `RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)`
  (hyperparameters carried over unchanged from the prior placeholder; **not
  claimed to be tuned** — see Verification Report Part 10)
- **Verified aggregate metrics (this audit):** MAE = 58.641, RMSE = 125.161,
  R² = 0.8805, `TimeSeriesSplit(n_splits=5)` on the same real dataset.
- **Status:** genuinely trained on real solar telemetry as of this
  revision. Reproduce with:

  ```bash
  python scripts/train_model.py --input data/Plant_1_Merged.csv
  ```

- **Feature importance (final refit, impurity-based):**

  | Feature | Importance |
  |---|---|
  | IRRADIATION | 0.8671 |
  | AC_POWER_LAG1 | 0.0842 |
  | Hour | 0.0193 |
  | AMBIENT_TEMPERATURE | 0.0147 |
  | MODULE_TEMPERATURE | 0.0146 |

  (`results/deployment_training.json`). Not to be confused with the
  research model's feature importance above — these are two separate
  models over two separate feature sets.

**Do not report the research model's MAE/RMSE/R² as belonging to the
deployment model** — they are two different feature sets, evaluated
separately, and happen to land in a similar R² range by coincidence of this
one dataset, not because they are the same experiment.

---

## Feature Engineering — What's Measured vs. Estimated

| Feature | Source at inference time | Notes |
|---|---|---|
| `IRRADIATION` | Open-Meteo `shortwave_radiation` (W/m²) ÷ 1000, clipped to [0, 1.3] | Confirmed against the real training data's `IRRADIATION` column (kW/m² scale, observed max 1.222). Clip widened from 1.2 → 1.3 since 1.2 sat inside the training range. |
| `Hour` | Local hour (0–23) at the selected location (Open-Meteo `timezone=auto`) | Directly comparable to training's `DATE_TIME.dt.hour` (plant-local time). |
| `MODULE_TEMPERATURE` | Heuristic estimate: `ambient_temperature + (shortwave_radiation / 800) * 20`, clipped to [-10, 85] °C | **Now also used at training time** (`scripts/train_model.py`), replacing the dataset's *measured* sensor value, so training and inference see the same feature distribution. The estimator's error against the real measured sensor: bias +0.13 °C, MAE 2.10 °C, RMSE 2.67 °C (systematically under-predicts the module-ambient gap at high irradiance and cannot reproduce the small night-time module-below-ambient offset — see Verification Report Part 7). |
| `AMBIENT_TEMPERATURE` | Open-Meteo live temperature | Directly comparable to the training column of the same name. |
| `AC_POWER_LAG1` | **Recursive/session-based, not measured telemetry, at inference time.** In the Streamlit app, this is either `0.0` (first prediction in a session for a given location) or the *model's own previous prediction* for that location, stored in `st.session_state`. | During training, `AC_POWER_LAG1` is the true `AC_POWER.shift(1)` (the actual previous measured reading). This is a genuine, disclosed train/inference mismatch — the live app has no measured AC power feed to draw on, so it substitutes its own last prediction. Called out in the UI and in Section 20 of the paper. |

---

## Model-Consistency Score (formerly labeled "Confidence")

`utils/confidence.py` computes:

```python
tree_predictions = [tree.predict(features)[0] for tree in model.estimators_]
consistency_score = 100 - (std(tree_predictions) / max_prediction_range) * 100
```

This measures how much the Random Forest's individual trees **agree with
each other** on a given input. It is:

- **Not** a calibrated probability
- **Not** a statistical confidence interval
- **Not** a measure of accuracy

A high score means the trees agree with each other; it says nothing about
whether the agreed-upon value is close to the true AC power. It has not
been calibrated against held-out real data. The UI labels this
"Model-Consistency Score" rather than "Confidence" for this reason.

---

## Illustrative Intraday Simulation

The dashboard's 24-hour curve is **not a validated forecast**. It:

- Scales the *current* weather reading up/down across a synthetic daylight
  bell curve.
- Recursively chains each hour's prediction into the next hour's
  `AC_POWER_LAG1` feature (compounding any error).
- Has never been evaluated against actual historical 24-hour observations.

The UI explicitly labels this "Illustrative Intraday Simulation."

---

## Installation

### 1. Clone and set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model file

`model/best_random_forest.pkl` is committed to this repo and, as of this
revision, is a genuinely trained model (see Verification Report). To
retrain it from scratch against the real dataset:

```bash
python scripts/train_model.py --input data/Plant_1_Merged.csv
```

For a UI-only synthetic demo model instead (never used by `app.py` itself):

```bash
python scripts/generate_demo_placeholder_model.py
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Run the tests

```bash
pytest tests/test_app.py -v
```

---

## Folder Structure

```
Solar-Prediction/
├── app.py                                   # Main Streamlit dashboard
├── data/
│   └── Plant_1_Merged.csv                    # Real merged dataset used for training/reproduction
├── model/
│   └── best_random_forest.pkl                # Genuinely trained (see Verification Report)
├── utils/
│   ├── weather.py                            # Open-Meteo geocoding + live weather
│   ├── feature_engineering.py                # Irradiation & module-temp derivation (train + inference)
│   ├── prediction.py                         # Local-only model loading + prediction + placeholder-detection heuristic
│   ├── confidence.py                         # Tree-disagreement-based model-consistency score
│   └── helpers.py                            # Logging, CSS, formatting, weather codes
├── scripts/
│   ├── generate_demo_placeholder_model.py     # Demo-only synthetic model generator (never used by the app)
│   ├── reproduce_research.py                  # Clean-process reproduction of the 2-feature research experiment
│   └── train_model.py                         # Trains the real 5-feature deployment model
├── results/
│   ├── research_reproduction.json             # Fold-by-fold + aggregate research metrics (this audit)
│   ├── research_feature_importance.csv
│   ├── deployment_training.json               # Fold-by-fold + aggregate deployment metrics (this audit)
│   └── figures/                               # Fresh, real figures generated from the above
├── tests/
│   └── test_app.py                            # Offline pipeline tests (12/12 passing)
├── docs/
│   └── VERIFICATION_REPORT.md                 # Full audit trail, all 25 parts of the brief, all numbers
├── eda.ipynb                                  # Original research notebook (two-feature experiment)
├── check_model.py                             # Small script to print a .pkl's metadata
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## Limitations

- **The originally claimed research metrics (46.959 / 98.166 / 0.9092) do
  not reproduce** and were replaced with freshly computed values (65.025 /
  141.608 / 0.8420) — see Verification Report Part 3.
- **Single-inverter, single-month historical research experiment**
  (`1BY6WEcLGh8j5v7`, 2020-05-15 to 2020-06-17) — results do not
  necessarily generalize to other inverters, plants, seasons, or
  geographies.
- **`MODULE_TEMPERATURE` is a heuristic estimate at inference time and is
  now also used (identically) at training time** — this closes the
  previous train/inference mismatch, but the estimate itself still departs
  from the true sensor value by a few °C (see table above).
- **`AC_POWER_LAG1` at inference time is a recursive model prediction, not
  a measured reading** — a real, disclosed train/inference mismatch.
- **The intraday 24-hour curve is illustrative only**, not a certified
  hour-by-hour forecast, and has not been evaluated against real 24-hour
  observations.
- **The model-consistency score is uncalibrated** — it is not a confidence
  interval or probability.
- **MAPE is unreliable on this dataset** due to zero/night-time AC power
  values and is intentionally not reported as a headline metric.
- **Live inference depends on Open-Meteo**, whose weather estimates
  (especially `shortwave_radiation`) may differ systematically from the
  historical plant's own sensor readings used at training time (potential
  distribution shift not directly measurable without live plant telemetry
  to compare against).
- **Only one inverter, one plant, one month of data** — no cross-plant or
  cross-season validation has been performed (Plant 2's generation/weather
  data was supplied alongside Plant 1's for this audit but was not used for
  training, since the deployed model, notebook, and paper all specify
  Plant 1 / inverter `1BY6WEcLGh8j5v7`).

---

## API Details

- **Provider:** [Open-Meteo](https://open-meteo.com) — free, no API key required.
- **Geocoding endpoint:** `https://geocoding-api.open-meteo.com/v1/search`
- **Forecast endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Fields used:** `temperature_2m`, `apparent_temperature`,
  `relative_humidity_2m`, `cloud_cover`, `wind_speed_10m`, `weather_code`,
  `shortwave_radiation`.
- Requests time out after 8 seconds and raise specific exceptions
  (`CityNotFoundError`, `WeatherAPITimeoutError`, `WeatherDataUnavailableError`)
  that the UI translates into friendly messages.

---

## Deployment Notes

Single-container Streamlit app, no database. Deploys to Streamlit Community
Cloud, Railway, Render, Fly.io, etc. with the start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

The app was smoke-tested in this audit (`streamlit run app.py --server.headless
true`, confirmed HTTP 200 on the local port) but has not been checked
against a live Streamlit Community Cloud deployment in this pass.

---

## License

This project is provided as-is for internal or educational use. Adapt the
license section to match your organization's requirements before
distribution.
