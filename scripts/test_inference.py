#!/usr/bin/env python3
# 2025-08-13 07:56:13 UTC - Doanduydong2

import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import mlflow
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = (PROJECT_ROOT / "app" / "app.py").resolve()
DATA_PATH = (PROJECT_ROOT / "data" / "bts_processed" / "flights_month_origin_carrier_base.parquet").resolve()
MODEL_PATH = (PROJECT_ROOT / "saved_models" / "final_best_flight_delay_model.joblib").resolve()
PREPROCESSOR_PATH = (PROJECT_ROOT / "saved_models" / "data_preprocessor_for_best_model.joblib").resolve()
CONFIG_PATH = (PROJECT_ROOT / "saved_models" / "preprocessor_feature_config_best_model.json").resolve()
OUTPUT_PATH = (PROJECT_ROOT / "reports" / "metrics" / "inference_benchmark.json").resolve()


def validate_paths():
    print("Validating required paths...")

    if not APP_PATH.exists():
        print(f"ERROR: app.py not found at {APP_PATH}")
        print("Make sure you're running the script from the project root directory.")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Make sure you've run scripts/make_dataset.py and scripts/make_features.py")
        sys.exit(1)

    if not all([MODEL_PATH.exists(), PREPROCESSOR_PATH.exists(), CONFIG_PATH.exists()]):
        print(f"ERROR: Model artifacts not found at {MODEL_PATH.parent}")
        print(f"  Model: {'Found' if MODEL_PATH.exists() else 'MISSING'}")
        print(f"  Preprocessor: {'Found' if PREPROCESSOR_PATH.exists() else 'MISSING'}")
        print(f"  Config: {'Found' if CONFIG_PATH.exists() else 'MISSING'}")
        print("\nPlease run scripts/train.py to train the model first:")
        print("  python scripts/train.py")
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("All required paths validated.")


def load_latest_sample(data_path):
    try:
        print(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)

        latest_period = df['year_month'].max()
        print(f"Latest time period in data: {latest_period}")

        latest_data = df[df['year_month'] == latest_period]

        if latest_data.empty:
            print("ERROR: No data found in the latest period")
            sys.exit(1)

        sample = latest_data.sample(1, random_state=42).iloc[0]
        return sample

    except Exception as e:
        print(f"ERROR loading data from {data_path}: {str(e)}")
        sys.exit(1)


def run_app_prediction(sample):
    try:
        required_features = [
            "airport", "arr_cancelled", "arr_diverted", "arr_flights", "carrier",
            "carrier_ct_rate", "late_aircraft_ct_rate", "month", "nas_ct_rate",
            "security_ct_rate", "weather_ct_rate"
        ]

        print(f"Using app.py at: {APP_PATH}")

        cmd = [sys.executable, str(APP_PATH)]

        for col in required_features:
            if col in sample and pd.notna(sample[col]):
                value = str(sample[col])
                if col == "month" and not value.isdigit():
                    value = str(int(float(value)))
            else:
                if col in ["arr_flights", "arr_cancelled", "arr_diverted",
                           "carrier_ct_rate", "late_aircraft_ct_rate",
                           "nas_ct_rate", "security_ct_rate", "weather_ct_rate"]:
                    value = "0.0"
                else:
                    value = ""

            cmd.extend([f"--{col}", value])

        print("\nRunning app.py with the following command:")
        print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR running app.py (return code {result.returncode}):")
            print(f"STDERR: {result.stderr}")
            return None

        return result.stdout

    except Exception as e:
        print(f"ERROR running app.py prediction: {str(e)}")
        return None


def load_model_artifacts():
    try:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
        print(f"Loading config from: {CONFIG_PATH}")

        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        with open(CONFIG_PATH, 'r') as f:
            feature_config = json.load(f)

        print("Model artifacts loaded successfully.")
        return model, preprocessor, feature_config

    except Exception as e:
        print(f"ERROR loading model artifacts: {str(e)}")
        sys.exit(1)


def prepare_input_dataframe(sample, feature_config):
    try:
        ordered_columns = feature_config["order_of_columns_in_X_train"]
        numeric_features = feature_config["numeric_features_for_pipe"]

        input_data = {}
        for col in ordered_columns:
            if col in sample and pd.notna(sample[col]):
                input_data[col] = sample[col]
            else:
                input_data[col] = 0.0 if col in numeric_features else ""

        input_df = pd.DataFrame([input_data])

        for col in numeric_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)

        return input_df

    except Exception as e:
        print(f"ERROR preparing input DataFrame: {str(e)}")
        sys.exit(1)


def benchmark_inference(model, preprocessor, input_df, n_warmup=10, n_runs=1000):
    try:
        print(f"Performing {n_warmup} warm-up runs...")
        for _ in range(n_warmup):
            X_transformed = preprocessor.transform(input_df)
            _ = model.predict(X_transformed)

        print(f"Measuring latency over {n_runs} runs...")
        latencies = []

        for _ in range(n_runs):
            start_time = time.perf_counter()
            X_transformed = preprocessor.transform(input_df)
            _ = model.predict(X_transformed)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        results = {
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "latency_p99_ms": float(p99),
            "n_runs": n_runs,
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies))
        }

        return results

    except Exception as e:
        print(f"ERROR benchmarking inference: {str(e)}")
        sys.exit(1)


def save_benchmark_results(results, output_path):
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Benchmark results saved to {output_path}")

    except Exception as e:
        print(f"ERROR saving benchmark results: {str(e)}")
        sys.exit(1)


def log_to_mlflow(results, sample, feature_config):
    try:
        mlflow.set_experiment("inference_benchmark")

        with mlflow.start_run(run_name="inference_latency_test"):
            mlflow.log_metric("latency_p50_ms", results["latency_p50_ms"])
            mlflow.log_metric("latency_p95_ms", results["latency_p95_ms"])
            mlflow.log_metric("latency_p99_ms", results["latency_p99_ms"])
            mlflow.log_metric("min_latency_ms", results["min_latency_ms"])
            mlflow.log_metric("max_latency_ms", results["max_latency_ms"])
            mlflow.log_metric("mean_latency_ms", results["mean_latency_ms"])
            mlflow.log_metric("std_latency_ms", results["std_latency_ms"])
            mlflow.log_metric("n_runs", results["n_runs"])

            ordered_columns = feature_config["order_of_columns_in_X_train"]
            for col in ordered_columns:
                if col in sample and pd.notna(sample[col]):
                    mlflow.log_param(f"sample_{col}", sample[col])

            if OUTPUT_PATH.exists():
                mlflow.log_artifact(str(OUTPUT_PATH))

            print("Results logged to MLflow experiment: inference_benchmark")

    except Exception as e:
        print(f"WARNING: Could not log to MLflow: {str(e)}")


def main():
    print("Flight Delay Model Inference Testing and Benchmarking")
    print("====================================================")

    validate_paths()

    print("\n1. Loading latest sample from processed data...")
    sample = load_latest_sample(DATA_PATH)

    print("\nSample data for inference:")
    for col, val in sample.items():
        if col in ["year_month", "airport", "carrier", "month", "arr_flights",
                   "delay_rate", "arr_del15", "arr_cancelled", "arr_diverted"]:
            print(f"  {col}: {val}")

    print("\n2. Running prediction with app.py...")
    prediction_output = run_app_prediction(sample)
    if prediction_output:
        print("\nPrediction result from app.py:")
        print(prediction_output)

    print("\n3. Loading model artifacts...")
    model, preprocessor, feature_config = load_model_artifacts()

    print("\n4. Preparing input data for benchmarking...")
    input_df = prepare_input_dataframe(sample, feature_config)

    print("\n5. Benchmarking inference latency...")
    benchmark_results = benchmark_inference(
        model,
        preprocessor,
        input_df,
        n_warmup=10,
        n_runs=1000
    )

    print("\nInference Latency Benchmark Results:")
    print(f"  P50: {benchmark_results['latency_p50_ms']:.2f} ms")
    print(f"  P95: {benchmark_results['latency_p95_ms']:.2f} ms")
    print(f"  P99: {benchmark_results['latency_p99_ms']:.2f} ms")
    print(f"  Min: {benchmark_results['min_latency_ms']:.2f} ms")
    print(f"  Max: {benchmark_results['max_latency_ms']:.2f} ms")
    print(f"  Mean: {benchmark_results['mean_latency_ms']:.2f} ms")
    print(f"  Std: {benchmark_results['std_latency_ms']:.2f} ms")

    print("\n6. Saving benchmark results...")
    save_benchmark_results(benchmark_results, OUTPUT_PATH)

    print("\n7. Logging to MLflow...")
    try:
        log_to_mlflow(benchmark_results, sample, feature_config)
    except Exception as e:
        print(f"MLflow logging skipped: {str(e)}")

    print("\nInference testing completed successfully!")


if __name__ == "__main__":
    main()