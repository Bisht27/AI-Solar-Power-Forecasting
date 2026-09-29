# Solar Power Forecasting — Verification Report

This report documents a from-scratch audit of the Solar-Prediction
repository, the delivered solar dataset, and the associated research paper.
Every number below was produced by actually executing code in this
environment (Python 3.12.3, package versions pinned to `requirements.txt`:
scikit-learn 1.5.2, pandas 2.2.2, numpy 1.26.4, joblib 1.4.2, streamlit
1.38.0). Nothing here is copied from prior notebook output except where
explicitly quoted and labeled as "as saved in the notebook."

---

## Part 1–2: Audit and Dataset Verification

**Files supplied for this pass:** `Solar-Prediction-audited.zip` (the
repository, already containing an earlier audit's `[DATA REQUIRED]` notes),
`Plant_1_Generation_Data.csv.zip`, `Plant_1_Weather_Sensor_Data.csv`,
`Plant_2_Generation_Data.csv.zip`, `Plant_2_Weather_Sensor_Data.csv`, and the
research paper `.docx`. Unlike the previous audit pass, the raw Plant 1/2
CSVs (from the well-known Kaggle "Solar Power Generation Data" set) were
available this time, so `Plant_1_Merged.csv` could be reconstructed and
every downstream claim actually re-run.

**Reconstruction:** `Plant_1_Generation_Data.csv` (68,778 rows, 22
inverters, `DATE_TIME` format `DD-MM-YYYY HH:MM`) was left-joined to
`Plant_1_Weather_Sensor_Data.csv` (3,182 rows, one weather sensor) on
`(DATE_TIME, PLANT_ID)`. Result: **68,778 rows, 11 columns** before adding
the notebook's engineered `Hour/Day/Month/Weekday` columns (15 columns
total once added, matching the notebook's `df.info()` output exactly).

| Claim (previously stated) | Verified |
|---|---|
| 68,778 original records | **Confirmed** (`len(df) == 68778`) |
| 15 columns | **Confirmed**, after adding `Hour/Day/Month/Weekday` to the 11 merge columns |
| 2020-05-15 to 2020-06-17 | **Confirmed** (`DATE_TIME.min()/.max()`) |
| Selected inverter `1BY6WEcLGh8j5v7` | **Confirmed present**, 3,154 rows for this inverter |
| 3,154 samples after filtering | **Confirmed exactly** |
| 15-minute sampling | **Confirmed for 3,143 of 3,154 consecutive intervals**; 11 non-15-minute gaps exist (see below) |

**Missing values:** 4 generation rows have no matching weather row (left
join leaves `NaN` for `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`,
`IRRADIATION`, `SOURCE_KEY_y` on those 4 rows) — exactly matching the
notebook's own `df.info()` output showing 68,774 non-null vs 68,778 total
for those columns. None of the 4 affected rows belong to the selected
inverter `1BY6WEcLGh8j5v7`, so they do not affect the research or
deployment experiments below. **How the notebook itself handled these 4
rows is not shown in the saved cells**, but the 3,154-row inverter subset
used downstream is unaffected either way.

**Duplicate timestamps:** none — 0 full-row duplicates, 0 duplicate
`(DATE_TIME, SOURCE_KEY_x)` pairs, confirmed both on the full merged file
and on the filtered inverter subset (3,154 unique `DATE_TIME` values).

**Gaps in the filtered inverter's time series:** 3,154 rows span
2020-05-15 00:00 to 2020-06-17 23:45, but only 3,143 of the 3,153
consecutive steps are exactly 15 minutes. 11 steps are irregular (listed in
full in `results/deployment_training.json`'s source run log), ranging from
30 minutes to 9 hours — real sensor/telemetry gaps in the underlying
Kaggle data, not an artifact of the merge. This does not violate the
"no shuffling, strictly chronological" requirement (the data is still
sorted and each `shift()` still only looks at the truly-adjacent row in
the DataFrame), but it does mean a handful of "next interval" targets are
actually several hours ahead rather than 15 minutes ahead. This was true
of the original notebook's pipeline too (it does not handle these gaps
specially), so this is a pre-existing property of the dataset, not
something introduced here.

**Units confirmed:** `AC_POWER` max 1,300.17 (kW), `DC_POWER` max
13,335.14 (matching the roughly 10:1 AC:DC ratio typical of this Kaggle
dataset), `IRRADIATION` max 1.222 (kW/m², i.e. already normalized, not raw
W/m²), `MODULE_TEMPERATURE`/`AMBIENT_TEMPERATURE` in °C.

---

## Part 3: Reproducing the Original Research Experiment (Clean Process)

`scripts/reproduce_research.py` was written to do this without touching or
trusting any saved notebook variable — it loads the CSV, filters/sorts,
builds `Target = AC_POWER.shift(-1)`, drops the resulting NaN row, selects
`features = ["IRRADIATION", "DAILY_YIELD"]`, and runs
`TimeSeriesSplit(n_splits=5)` with a fresh `RandomForestRegressor
(n_estimators=100, random_state=42, n_jobs=-1)` fit per fold — exactly the
steps visible in the notebook's code cells. A first pass (3,153 usable
samples, dropping only the single row whose `Target` is `NaN`) came within
noise of, but did not exactly match, the fold-level numbers already visible
in the notebook's own saved output. Trimming exactly one additional
trailing row before constructing the target (3,152 usable samples) then
**reproduced the notebook's own previously-saved Fold 21/22 MAE, RMSE and
R² bit-for-bit**, confirming the training/validation code itself is
correct and deterministic, and that the notebook's `backdata.csv` most
likely had one fewer usable row than this audit's reconstructed
`Plant_1_Merged.csv` (e.g. a marginally different join, or one additional
row dropped upstream that this audit's join did not need to drop). The
table below uses this exact-match convention and renumbers the folds 1–5
(the notebook's own labels, "Fold 21" through "Fold 25", are addressed
under Root Cause below).

**Fold-by-fold results (this audit, clean run — exact match to the
notebook's own saved Fold 21/22 output):**

| Fold | Train | Test | MAE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | 527 | 525 | 68.373 | 134.159 | 0.8961 |
| 2 | 1,052 | 525 | 60.515 | 120.592 | 0.8978 |
| 3 | 1,577 | 525 | 61.121 | 122.026 | 0.8877 |
| 4 | 2,102 | 525 | 62.194 | 157.878 | 0.7928 |
| 5 | 2,627 | 525 | 72.922 | 173.386 | 0.7354 |
| **Mean** | | | **65.025** | **141.608** | **0.8420** |

(Folds 1–2 above are an exact, digit-for-digit match to the values the
paper had previously published for "Fold 21" and "Fold 22"; Folds 3–5 are
newly confirmed numeric values for what the paper had previously left as
`[DATA REQUIRED]` for "Fold 23–25".)

**Comparison with the claimed aggregate (MAE 46.959, RMSE 98.166,
R² 0.9092): it does not match**, and the discrepancy is not explained by a
rounding or sample-count difference. Two independent checks confirm this:

1. **The notebook's own saved per-fold output** (`eda.ipynb`, the cell
   printing "BACKTEST FOLD 21" through "FOLD 25") shows individual fold
   values whose **mean is MAE 65.025 / RMSE 141.608 / R² 0.8420** — exactly
   what this audit's clean re-run also produces (see table above), but
   **not the claimed aggregate of 46.959 / 98.166 / 0.9092 that the same
   notebook prints two cells later.** In other words, the notebook's own
   two adjacent outputs are mutually inconsistent, and this audit's
   from-scratch reproduction sides with the fold-level numbers, which are
   internally consistent and independently reproducible, over the
   headline aggregate, which is neither.
2. Re-running the experiment with several other plausible feature/target
   variations (adding `DC_POWER`, `Hour`, `AC_POWER_LAG1`, `MODULE_
   TEMPERATURE`, `AMBIENT_TEMPERATURE`, `TOTAL_YIELD`, individually and in
   combination) was tried as a forensic check for what configuration could
   have produced 46.959/98.166/0.9092. None reproduced it (closest found:
   `IRRADIATION + DC_POWER + DAILY_YIELD` → MAE 56.238, R² 0.9014 — still
   off, and includes a feature not in the documented experiment).

**Root cause (most likely):** the notebook's fold-numbering itself is a
clue — the printed folds are labeled 21–25, not 1–5, and `fold` is never
initialized in any surviving cell (`fold=` does not appear as an
assignment anywhere in the saved notebook, nor do `tscv`, `mae_scores`,
`rmse_scores`, or `r2_scores`). This means the training/backtesting cell
was executed roughly 21+ times across a live kernel session whose counters
(`fold`, the score-accumulator lists) were **never reset**, so the "FINAL
BACKTEST RESULTS" cell's `np.mean(mae_scores)` etc. is averaging over all
historical accumulated runs from that session (very possibly using
different feature sets on earlier iterations), not just the 5 folds whose
individual numbers happen to still be visible in the saved output. This
is a classic Jupyter hidden-state bug, not evidence of fabrication — but it
does mean **46.959/98.166/0.9092 cannot be trusted or reproduced** and per
the audit brief's explicit instruction, **the newly reproduced values
(65.025 / 141.608 / 0.8420) are used instead** in this repository and its
paper going forward.

---

## Part 4: The Stale `DC_POWER` Feature-Importance Result

Confirmed and root-caused via the notebook's own execution-order metadata
(`cell.execution_count`, which increments every time a cell is *run*, even
if its saved output is from an earlier run of a *different* cell):

| Cell (by position) | Execution # | Content |
|---|---|---|
| 21 | 50 | `df = pd.read_csv("backdata.csv")` ... filter to inverter → 3,154 rows |
| 22 | 51 | Drop `SOURCE_KEY_x/y`, `PLANT_ID` |
| **27** | **65** | `final_model.feature_importances_` → prints `DC_POWER` at ≈1.0 |
| 25 | 67 | The `tscv.split(X)` backtesting loop (folds 21–25) |
| 26 | 68 | "FINAL BACKTEST RESULTS" → prints 46.959/98.166/0.9092 |
| **28** | **71** | `final_model.feature_names_in_` → prints `['IRRADIATION' 'DC_POWER' 'DAILY_YIELD']` |
| 23 | 76 | `features = ["IRRADIATION", "DAILY_YIELD"]` (2-feature, redefined **after** cell 27/28 executed) |
| 24 | 78 | Confirms `X.columns == ['IRRADIATION', 'DAILY_YIELD']` |

**`final_model` is never assigned in any surviving cell** — its definition
was executed and then either overwritten or its defining cell was deleted
before the notebook was saved. Its `feature_names_in_`
(`['IRRADIATION', 'DC_POWER', 'DAILY_YIELD']`, 3 features) does **not**
match the 2-feature `X` (`IRRADIATION`, `DAILY_YIELD`) used in the
backtesting loop that produced the reported MAE/RMSE/R². Worse, the
execution-count ordering shows `features` was **redefined to 2 columns
after** the feature-importance cell already ran against the stale
3-feature `final_model`.

**Conclusion: the `DC_POWER`-dominated feature-importance chart belongs to
a different, earlier, unlogged model configuration and must not be used to
describe the evaluated 2-feature research experiment.** A fresh
feature-importance table was generated in this audit by refitting a
`RandomForestRegressor` with the exact same hyperparameters on the exact
same `X = df[["IRRADIATION", "DAILY_YIELD"]]`, `y = df["Target"]` used for
the (now correctly reproduced) backtesting:

| Feature | Importance (this audit, research model) |
|---|---|
| IRRADIATION | 0.9295 |
| DAILY_YIELD | 0.0705 |

See `results/research_feature_importance.csv` and
`results/figures/research_feature_importance.png`. This is clearly
distinguished in the paper (Section 14.2) from the deployment model's
feature importance (Section 23 / Table below).

---

## Part 5–8: Deployment Model — Feature Feasibility, Lag, Module Temp, Irradiation Scale

All five deployment features (`IRRADIATION`, `Hour`, `MODULE_TEMPERATURE`,
`AMBIENT_TEMPERATURE`, `AC_POWER_LAG1`) **can** legitimately be constructed
from `Plant_1_Merged.csv` for the same inverter used in the research
experiment. Specifics:

- **`AC_POWER_LAG1 = AC_POWER.shift(1)`**, computed after sorting the
  filtered inverter's rows chronologically by `DATE_TIME`, so `shift(1)`
  can only ever pull the immediately preceding row *of the same inverter*,
  strictly earlier in time. The very first row (no predecessor) gets
  `NaN` and is dropped along with the very last row (`Target` is `NaN`
  there) — exactly 2 rows dropped total, confirmed by the training script's
  own printed row-count check.
- **`MODULE_TEMPERATURE` is measured in the raw dataset** (a real weather-
  sensor reading, e.g. 22.86 °C in the very first row) but Open-Meteo,
  used at live-inference time, has no module-temperature field — the
  deployment code has always estimated it heuristically:
  `Ambient + (Shortwave Radiation / 800) * 20`. **Before this audit, the
  training script trained on the *measured* sensor value while inference
  only ever sees the *heuristic estimate* — a genuine train/inference
  distribution mismatch.** This was measured directly:
  - Correlation between heuristic estimate and measured value: 0.9885
    (close, but not identical)
  - Bias: +0.14 °C; MAE: 2.10 °C; RMSE: 2.67 °C (over all 3,152 usable
    rows)
  - At high irradiance (`IRRADIATION > 0.5`), the measured
    module-minus-ambient gap averages 20.70 °C while the heuristic only
    predicts an 18.05 °C gap (mean error −2.65 °C) — the heuristic
    systematically **under-estimates** module heating in bright conditions.
  - At night (`IRRADIATION == 0`), the measured sensor shows the module
    running **1.82 °C below** ambient on average (plausible: radiative
    cooling), which the heuristic formula (which returns exactly the
    ambient temperature at zero radiation) cannot reproduce at all.
  - **Fix applied:** `scripts/train_model.py` now computes
    `MODULE_TEMPERATURE` at training time with the *identical* function
    (`utils.feature_engineering.estimate_module_temperature`) the
    deployed app calls at inference, fed with the row's own `IRRADIATION`
    converted back to W/m² (`irradiation_to_shortwave_radiation`). The
    measured sensor value is retained only as `MODULE_TEMPERATURE_MEASURED`
    for the diagnostic comparison above — it is never used as a training
    feature. This does not make the estimator itself more accurate, but it
    does mean the model is trained on the same feature distribution it
    will see in production, rather than a systematically different one.
- **`IRRADIATION` scale confirmed compatible.** Training data's
  `IRRADIATION` ranges 0.000–1.222 (kW/m², i.e. a fraction of the ~1,000
  W/m² "standard test condition" irradiance) with hour-of-day peaks
  matching a normal solar noon curve (peak ≈1.0–1.2 around 12:00–14:00,
  zero outside 05:00–19:00). Open-Meteo's `shortwave_radiation` is in
  W/m², so `IRRADIATION = shortwave_radiation / 1000` (the pre-existing
  transformation) is the **correct** unit conversion — confirmed, not
  assumed. One change was made: the upper clip was **1.2**, which sits
  *inside* the observed training maximum (1.222) and was silently
  truncating a small number of legitimate near-peak readings; it is now
  **1.3**. Because a Random Forest cannot meaningfully extrapolate past
  its training range regardless of the input clip, this only changes what
  is *displayed*, not what the model can predict for inputs already above
  the training range.

---

## Part 9: Deployment Model Training and Validation

`scripts/train_model.py`, run against `data/Plant_1_Merged.csv`, inverter
`1BY6WEcLGh8j5v7`, 3,152 usable rows (2 dropped for lag/target NaNs, same
as the research experiment):

| Fold | Train | Test | MAE | RMSE | R² |
|---|---|---|---|---|---|
| 1 | 527 | 525 | 64.156 | 124.920 | 0.9099 |
| 2 | 1,052 | 525 | 55.445 | 116.419 | 0.9054 |
| 3 | 1,577 | 525 | 55.705 | 111.728 | 0.9051 |
| 4 | 2,102 | 525 | 58.626 | 141.992 | 0.8333 |
| 5 | 2,627 | 525 | 59.275 | 130.746 | 0.8486 |
| **Mean** | | | **58.641** | **125.161** | **0.8805** |

Conventional MAPE was not computed, for the same reason as the research
experiment (many zero-valued nighttime targets make it numerically
unstable).

---

## Part 10: Hyperparameters

`n_estimators=200, max_depth=12, random_state=42, n_jobs=-1` — carried over
unchanged from the previously-committed (placeholder) model and from
`scripts/generate_demo_placeholder_model.py`. **No hyperparameter tuning
was performed in this pass.** No search space, tuning validation procedure,
or selection metric is claimed, because none was run.

---

## Part 11: Data Leakage Audit

| Check | Result |
|---|---|
| Target shift | `Target = AC_POWER.shift(-1)`, computed *after* sorting by `DATE_TIME` — always strictly the next chronological row. |
| Lag generation | `AC_POWER_LAG1 = AC_POWER.shift(1)`, same sorted frame — always strictly the previous chronological row. |
| Chronological splitting | `TimeSeriesSplit` used throughout; no `train_test_split`, no shuffling anywhere in either training script. |
| Duplicate timestamps | 0 found (checked above). |
| Future weather information | None used — all features (`IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`) are the *current* row's own sensor readings, not looked ahead. |
| Train/test contamination | Each `TimeSeriesSplit` fold's training window ends strictly before its test window begins (verified via the printed `train_end`/`test_start` timestamps — e.g. fold 1: train ends 2020-05-20 19:15, test starts 2020-05-20 19:30, exactly one 15-minute step later). |

**No leakage found** in either the research or deployment pipeline.

---

## Part 12–13: Serialization and Placeholder Risk

`model/best_random_forest.pkl` was overwritten by `scripts/train_model.py`
in this pass. Reloaded and verified:

```
Model type        : <class 'sklearn.ensemble._forest.RandomForestRegressor'>
n_features_in_    : 5
feature_names_in_ : ['IRRADIATION', 'Hour', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE', 'AC_POWER_LAG1']
Parameters        : {..., 'n_estimators': 200, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1, ...}
```

`utils.prediction.is_placeholder_model(model)` → **`False`** for the
retrained model (it was `True` for the file this repository shipped with
before this pass — confirmed directly by loading the original committed
`.pkl`, regenerating the synthetic-placeholder reference with its fixed
seed, and finding the two agree to within floating-point noise on 200
synthetic samples, and further confirmed by evaluating the original
placeholder against the *real* dataset: MAE 85.29, R² 0.8375 — much worse
than a model that has actually seen this data, and computed on features it
was never trained on).

`scripts/generate_demo_placeholder_model.py` remains, labeled DEMO ONLY /
SYNTHETIC DATA / NOT FOR RESEARCH / NOT FOR PRODUCTION in its own
docstring, and is never imported by `app.py` (confirmed by grep — no
`import` of that module anywhere outside itself and the test file's
placeholder-heuristic test).

---

## Part 14: Model Loading (Local File Only)

The application loads the deployment model only from the local file
`model/best_random_forest.pkl` (or `model/rf_model.pkl`) using `joblib`, in
`utils/prediction.py`. If neither file exists, `resolve_model_path()` raises
`ModelLoadError`; no model is downloaded from any source. `requirements.txt`
lists only the packages the application needs.

## Part 15–16: Streamlit Verification

- `streamlit run app.py --server.headless true` was actually started in
  this container; a request to the local port returned **HTTP 200**, and
  the app rendered without import or startup errors (log tail confirms
  `You can now view your Streamlit app`). Re-verified after the
  local-only loading change (Part 14) and the warnings fix (see the
  "Second Audit Pass" section): still HTTP 200.
- `get_model()` loads via `utils.prediction.load_model()`, which now finds
  the local `model/best_random_forest.pkl` **only** — there is no remote
  fallback (Part 14) — confirmed by reading
  `resolve_model_path()`'s candidate-path logic, which raises
  `ModelLoadError` directly if no local candidate exists.
- Feature order is hardcoded identically in three places
  (`utils/feature_engineering.py FEATURE_COLUMNS`,
  `utils/prediction.py EXPECTED_FEATURE_ORDER`,
  `tests/test_app.py EXPECTED_FEATURES`) and all three were confirmed
  textually identical.
- `app.py`'s "Model Information" panel calls `is_placeholder_model(model)`
  and shows a warning banner if `True` — it does **not** silently fall
  back to a placeholder without warning the user.
- `AC_POWER_LAG1` in `app.py`: initialized to `st.session_state.get(lag_key,
  0.0)` per `(latitude, longitude)` key, and after each prediction is
  overwritten with **that prediction itself**
  (`st.session_state[lag_key] = predicted_power`) — confirmed to be **case
  B** in the brief's terms (the app's own previous prediction), never
  actual measured plant telemetry. `app.py`'s own inline comment already
  says this explicitly; this audit did not need to change that behavior,
  only confirm it and ensure the paper describes it accurately (it did).

---

## Part 17–18: Intraday Simulation and Model-Consistency Score

- `simulate_intraday_curve()` scales the *current* single weather reading
  across a synthetic daylight bell curve and chains each hour's own
  prediction into the next hour's `AC_POWER_LAG1` — confirmed by reading
  the function body. `app.py` already labels this "Illustrative Intraday
  Simulation" in the UI and in an inline caption; no change was needed
  beyond confirming the label is accurate and that no 24-hour accuracy
  metric is claimed anywhere in the app or paper.
- `utils/confidence.py`'s `calculate_confidence()` computes
  `100 - (std(tree_predictions) / max_prediction_range) * 100` from
  `model.estimators_` (individual Random Forest trees) — a tree-
  disagreement measure, not a calibrated probability. Already labeled
  "Model-Consistency Score" throughout the app and code; confirmed
  accurate, no calibration step exists anywhere in the codebase.

---

## Part 19: Final Comparison (Research vs. Deployment)

| | Research Model | Deployment Model |
|---|---|---|
| Features | `IRRADIATION`, `DAILY_YIELD` | `IRRADIATION`, `Hour`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`, `AC_POWER_LAG1` |
| Model | `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` | `RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)` |
| Validation | `TimeSeriesSplit(n_splits=5)` | `TimeSeriesSplit(n_splits=5)` |
| MAE | **65.025** (reproduced; claimed 46.959 does not hold up) | **58.641** |
| RMSE | **141.608** | **125.161** |
| R² | **0.8420** | **0.8805** |

Both were evaluated on the same 3,152–3,153-row inverter subset of the same
dataset, but with different feature sets, different `n_estimators`/
`max_depth`, and — critically — the deployment model's features include a
lagged version of the target itself (`AC_POWER_LAG1`), which the research
model does not have access to. **This is not a controlled, apples-to-apples
comparison**, so this report does not claim the deployment model is
"better" than the research model in any general sense, only that both
numbers are now real, reproducible, and computed the same way
(`TimeSeriesSplit(n_splits=5)`, same underlying data, same inverter).

---

## Part 20–21: Repository Updates and Tests

**Files changed in this pass:**
- `utils/feature_engineering.py` — widened `IRRADIATION` clip 1.2→1.3;
  added `irradiation_to_shortwave_radiation()` and `hour_from_timestamp()`
  helpers; documented the verified scale/estimator numbers inline.
- `scripts/train_model.py` — rewritten to actually run (previous version
  only printed `[DATA REQUIRED]` because no dataset was available in the
  prior pass); trains `MODULE_TEMPERATURE` with the same estimator used at
  inference (train/inference consistency fix); writes
  `results/deployment_training.json`.
- `scripts/reproduce_research.py` — new; clean-process reproduction of the
  2-feature research experiment, independent of any notebook state; writes
  `results/research_reproduction.json` and
  `results/research_feature_importance.csv`.
- `model/best_random_forest.pkl` — retrained on real data (was a synthetic
  placeholder).
- `app.py` — fixed a stale script-name reference in the model-load-error
  caption (`generate_placeholder_model.py` → the real filename,
  `generate_demo_placeholder_model.py` / `train_model.py`).
- `tests/test_app.py` — updated the irradiation-clip assertion from 1.2 to
  1.3 to match the corrected clip constant.
- `data/Plant_1_Merged.csv` — added; the reconstructed merged dataset, so
  `scripts/train_model.py --input data/Plant_1_Merged.csv` and
  `scripts/reproduce_research.py --input data/Plant_1_Merged.csv` are
  runnable as documented.
- `README.md` — rewritten to reflect the completed audit and real numbers
  (previous version was written for the pre-data, placeholder-only state).
- `docs/VERIFICATION_REPORT.md` — this file.
- `results/` — new; all raw JSON/CSV outputs and the three figures
  referenced above and in the updated paper.

**Test results:** `pytest tests/test_app.py -v` → **12 passed, 0 failed**
(one test, `test_irradiation_and_module_temp_clipping`, needed its expected
value updated from 1.2 to 1.3 to match the corrected clip constant; after
that single change, all 12 pass). Verified line-by-line against the
brief's minimum checklist: model loads ✓, 5 features ✓, feature order ✓,
prediction works ✓, feature engineering works ✓, lag feature works ✓, model
is confirmed not the placeholder ✓ (`is_placeholder_model` → `False`),
application module parses/imports ✓, intraday simulation runs (24 rows) ✓,
weather-handling clipping ✓.

---

## Second Audit Pass: Local-Only Model Loading, Warnings Fix, Deployment Feature Importance

A follow-up pass addressed three items raised after the first pass:

**1. Local-only model loading.** The schema check lives in
`utils/prediction.py` as `check_feature_schema()` and runs inside
`load_model()` against the local file. `resolve_model_path()` raises
`ModelLoadError` if no local candidate exists and makes no network call.

**2. The 200 (later 600, see below) sklearn `UserWarning`s were root-caused
and fixed, not suppressed.** The warning
(`X has feature names, but DecisionTreeRegressor was fitted without
feature names`) came from `utils/confidence.py`'s
`calculate_confidence()`, which called `tree.predict(features_df)` — a
pandas DataFrame — separately on each of the Random Forest's individual
`estimators_`. Each individual tree is fit internally on plain NumPy
arrays (scikit-learn does not propagate the forest's column-name metadata
down to the sub-estimators), so predicting on a DataFrame at that level
triggers the warning on every tree. Fixed by converting to
`features_df.to_numpy()` once, outside the per-tree loop, before calling
`tree.predict()` — verified with a `warnings.catch_warnings(record=True)`
check to return exactly 0 warnings, with `calculate_confidence()`'s output
value unchanged (only the input array type to each tree changed, not the
computation itself). No global warning filter was used.

A second, unrelated, separately-discovered issue temporarily inflated the
apparent warning count to 600: an earlier `pip install matplotlib` in this
audit's own virtual environment had transitively upgraded `numpy` from the
pinned `1.26.4` to `2.5.3`, which caused `joblib.load()` to emit a
`DeprecationWarning` while deserializing the pickled model's internal
NumPy arrays (an unrelated NumPy-pickle-format issue, not a code bug in
this repository). This was caught by re-checking installed package
versions against `requirements.txt`'s pins and was fixed by reinstalling
the exact pinned versions in the audit environment
(`numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.2`, `joblib==1.4.2`).
After both fixes: **`pytest tests/test_app.py -v` → 12 passed, 0 warnings.**

**3. Schema/loading test.** `test_local_model_schema_check_and_no_remote_fallback`
verifies (a) `check_feature_schema()` flags a wrong-schema model and accepts a
correct-schema one, and (b) `resolve_model_path()` raises `ModelLoadError`
when no local file exists, with socket creation disabled so any network use
would fail the test. The test count is 12.

**4. Deployment model feature importance recorded.** From the same
retrained model referenced throughout this report
(`results/deployment_training.json`):

| Feature | Importance |
|---|---|
| IRRADIATION | 0.8671 |
| AC_POWER_LAG1 | 0.0842 |
| Hour | 0.0193 |
| AMBIENT_TEMPERATURE | 0.0147 |
| MODULE_TEMPERATURE | 0.0146 |

This is a separate result from the research model's feature importance
(Part 4 above, `IRRADIATION` ≈ 0.930, `DAILY_YIELD` ≈ 0.070) — two
different models over two different feature sets, and the two must not be
conflated or averaged together.

**Files changed in this second pass:** `utils/prediction.py` (local-only
loading + `check_feature_schema`), `utils/confidence.py` (NumPy-array fix for
the per-tree predict warning), `requirements.txt`, `tests/test_app.py`,
`app.py` (removed an unsupported "hyperparameter tuned" claim from the
sidebar; softened "Next 15 minutes" to "approximately the next 15 minutes"),
`README.md` (loading note, deployment feature-importance table, target-horizon
wording), and this file.

---

## Part 24: Reproducibility Commands

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest==8.3.3   # dev/test only

# 2. Dataset placement (already done in this repo at data/Plant_1_Merged.csv;
#    to rebuild from the raw Kaggle files instead):
python - <<'PY'
import pandas as pd
g = pd.read_csv("Plant_1_Generation_Data.csv")
w = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
g["DATE_TIME"] = pd.to_datetime(g["DATE_TIME"], format="%d-%m-%Y %H:%M")
w["DATE_TIME"] = pd.to_datetime(w["DATE_TIME"])
g.merge(w, on=["DATE_TIME", "PLANT_ID"], how="left").to_csv("data/Plant_1_Merged.csv", index=False)
PY

# 3. Research reproduction (clean process, no notebook state)
python scripts/reproduce_research.py --input data/Plant_1_Merged.csv

# 4. Deployment model training
python scripts/train_model.py --input data/Plant_1_Merged.csv

# 5. Testing
pytest tests/test_app.py -v

# 6. Streamlit execution
streamlit run app.py
```

All six commands above were actually executed in this container as part of
producing this report; none are hypothetical.

---

## Remaining Open Items (not resolved in this pass)

- The exact reason the notebook's kernel accumulated stale `fold`/score
  state (Part 3) cannot be recovered further without the original live
  kernel session — the saved `.ipynb` alone is enough to *detect and
  explain* the inconsistency, not to recover the specific earlier runs
  that produced 46.959/98.166/0.9092.
- No hyperparameter tuning, cross-inverter generalization test, or
  live-vs-actual accuracy comparison was performed — all are listed as
  future work in the paper's Section 21, consistent with the brief's
  instruction not to claim tuning that wasn't done.
