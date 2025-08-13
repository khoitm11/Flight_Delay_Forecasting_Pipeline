# Flight Delay Prediction

End-to-end system to predict monthly **arrival delay rate** for U.S. domestic flights. It ingests BTS on-time performance data, aggregates features, trains and tracks models with MLflow, and exposes a CLI for inference. Data source: **BTS On-Time Performance & Delay Causes**.

---

## Results (hold-out 2022-12 → 2023-08)

| Model                     |        MAE |       RMSE |         R² | Spearman ρ | ΔMAE vs baseline |
| ------------------------- | ---------: | ---------: | ---------: | ---------: | ---------------: |
| GradientBoostingRegressor | **0.0257** | **0.0366** | **0.9133** | **0.9588** |          −0.0770 |

**Inference latency (CPU, 1k runs):** p50 **3.62 ms**, p95 **4.93 ms**, p99 **6.04 ms**.

Evaluation uses **expanding-window TimeSeriesSplit** to avoid leakage when forecasting future months from past data.

---

## Data

* Source: U.S. DOT / BTS On-Time and delay causes. You can download monthly files or curated extracts.
* Processed training table produced by the repo:
  `data/bts_processed/flights_month_origin_carrier_base.parquet`
  Note: `pandas.read_parquet` requires **pyarrow** or **fastparquet** installed.

---

## Features

The training and inference interface uses **11 columns**:

```
airport, arr_cancelled, arr_diverted, arr_flights, carrier,
carrier_ct_rate, late_aircraft_ct_rate, month, nas_ct_rate,
security_ct_rate, weather_ct_rate
```

Preprocessing is built with `ColumnTransformer`: scale numeric features and one-hot encode categorical features. This is the canonical pattern for mixed-type tabular data.

---

## Repo layout

```
flight_delay_bts_project/
├─ app/
│  └─ app.py                      # CLI inference (stable 11-feature interface)
├─ scripts/
│  ├─ make_features.py            # build monthly aggregates → parquet
│  ├─ train.py                    # train with TimeSeriesSplit, log to MLflow, save artifacts
│  └─ test_inference.py           # run app.py once and benchmark latency
├─ data/
│  ├─ bts_raw/                    # place raw CSVs here (gitignored)
│  └─ bts_processed/              # generated parquet (gitignored)
├─ saved_models/
│  ├─ final_best_flight_delay_model.joblib
│  ├─ data_preprocessor_for_best_model.joblib
│  └─ preprocessor_feature_config_best_model.json
├─ mlruns/                        # MLflow Tracking on local file store
└─ reports/metrics/               # plots + benchmark JSON
```

---

## Quickstart

### 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Build features

Put your BTS CSV(s) in `data/bts_raw/`, then:

```bash
python scripts/make_features.py
# writes data/bts_processed/flights_month_origin_carrier_base.parquet
```

### 3) Train and log

```bash
python scripts/train.py
mlflow ui                        # open http://127.0.0.1:5000 to inspect runs
```

MLflow Tracking logs parameters, metrics, and artifacts. This repo pins the tracking URI to the project's `mlruns/` path for consistent local logging.

### 4) Inference via CLI

After training, ensure the three artifacts exist in `saved_models/`:

```bash
python app/app.py \
  --airport PHX --arr_cancelled 6 --arr_diverted 2 --arr_flights 456 --carrier AA \
  --carrier_ct_rate 0.1164 --late_aircraft_ct_rate 0.1573 --month 8 \
  --nas_ct_rate 0.0647 --security_ct_rate 0.0 --weather_ct_rate 0.0086
```

### 5) Benchmark latency

```bash
python scripts/test_inference.py
# prints p50/p95/p99 and saves reports/metrics/inference_benchmark.json
```

---

## Modeling notes

* Cross-validation uses **TimeSeriesSplit** with an expanding train window. Shuffling is disabled by design for time-ordered data.
* No lag features in the final interface to keep the CLI stable and stateless.
* Preprocessing uses `StandardScaler` for numeric and `OneHotEncoder(handle_unknown="ignore")` for categorical inside a single `ColumnTransformer` pipeline.

---

## Docker (runtime image)

* Minimal image that copies `app/` and `saved_models/`, installs `requirements.txt`, and exposes `ENTRYPOINT ["python", "app/app.py"]`.
* Build and run:

```bash
docker build -t flight-delay-infer .
docker run --rm flight-delay-infer --airport PHX --arr_cancelled 6 --arr_diverted 2 \
  --arr_flights 456 --carrier AA --carrier_ct_rate 0.1164 --late_aircraft_ct_rate 0.1573 \
  --month 8 --nas_ct_rate 0.0647 --security_ct_rate 0.0 --weather_ct_rate 0.0086
```

---

## Reproducibility

* Python ≥3.10.
* Deterministic seeds where supported by estimators.
* Local MLflow file store at `./mlruns` (change via `MLFLOW_TRACKING_URI` or code).

---

## Acknowledgements

* U.S. DOT / BTS for public on-time and delay cause data.
* scikit-learn for `TimeSeriesSplit`, pipelines, and `ColumnTransformer`.
* MLflow for experiment tracking and artifacts UI.
