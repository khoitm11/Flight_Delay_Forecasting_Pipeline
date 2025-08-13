#!/usr/bin/env python3
# 2025-08-13 07:58:20 UTC - Doanduydong2

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
import mlflow
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "metrics"

SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlruns'}")
mlflow.set_experiment("flight_delay_project_training")

FEATURE_COLUMNS = [
    "airport", "arr_cancelled", "arr_diverted", "arr_flights", "carrier",
    "carrier_ct_rate", "late_aircraft_ct_rate", "month", "nas_ct_rate",
    "security_ct_rate", "weather_ct_rate"
]

NUMERIC_FEATURES = [
    "arr_flights", "arr_cancelled", "arr_diverted", "carrier_ct_rate",
    "late_aircraft_ct_rate", "nas_ct_rate", "security_ct_rate", "weather_ct_rate"
]

CATEGORICAL_FEATURES = ["month", "carrier", "airport"]
TARGET_COLUMN = "delay_rate"


def create_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.unique(y_pred).size > 1:
        spearman_corr, _ = spearmanr(y_true, y_pred)
    else:
        spearman_corr = 0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'spearman': spearman_corr
    }


def create_plots(y_true, y_pred, title, model_name, fold_idx):
    plot_files = {}

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Delay Rate')
    plt.ylabel('Predicted Delay Rate')
    plt.title(f'{title} - {model_name}')
    plt.tight_layout()

    fig_path = REPORTS_DIR / f"actual_vs_pred_{model_name}_fold{fold_idx}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    plot_files['actual_vs_pred'] = str(fig_path)

    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residuals Distribution - {model_name}')
    plt.tight_layout()

    fig_path = REPORTS_DIR / f"residuals_{model_name}_fold{fold_idx}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    plot_files['residuals'] = str(fig_path)

    return plot_files


def persist_best_artifacts(best_model, final_preprocessor, final_feature_config):
    model_path = SAVED_MODELS_DIR / "final_best_flight_delay_model.joblib"
    preprocessor_path = SAVED_MODELS_DIR / "data_preprocessor_for_best_model.joblib"
    config_path = SAVED_MODELS_DIR / "preprocessor_feature_config_best_model.json"

    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path.resolve()}")

    joblib.dump(final_preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path.resolve()}")

    with open(config_path, 'w') as f:
        json.dump(final_feature_config, f, indent=2)
    print(f"Feature config saved to: {config_path.resolve()}")

    if not all([model_path.exists(), preprocessor_path.exists(), config_path.exists()]):
        missing = []
        if not model_path.exists(): missing.append("model")
        if not preprocessor_path.exists(): missing.append("preprocessor")
        if not config_path.exists(): missing.append("feature config")

        raise FileNotFoundError(f"Failed to save required artifacts: {', '.join(missing)}")

    print(f"Saved all artifacts to: {SAVED_MODELS_DIR.resolve()}")

    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "config_path": str(config_path)
    }


def train_and_evaluate_models(X, y, time_series_splits, feature_config):
    print("Training and evaluating models...")

    models = {
        'DummyRegressor': DummyRegressor(strategy='mean'),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }

    model_results = {name: {'fold_metrics': []} for name in models.keys()}

    last_fold = time_series_splits[-1]
    holdout_train_idx, holdout_test_idx = last_fold

    X_with_idx = X.reset_index(drop=True)
    y_with_idx = y.reset_index(drop=True)

    with mlflow.start_run(run_name="training_pipeline"):
        mlflow.log_param("n_splits", len(time_series_splits))
        mlflow.log_param("feature_columns", feature_config['order_of_columns_in_X_train'])
        mlflow.log_param("numeric_features", feature_config['numeric_features_for_pipe'])
        mlflow.log_param("categorical_features", feature_config['categorical_features_for_pipe'])
        mlflow.log_param("cv_strategy", "time-series CV (expanding window, TimeSeriesSplit)")

        mlflow.log_param("python_version", os.popen('python --version').read().strip())
        mlflow.log_param("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        for fold_idx, (train_idx, test_idx) in enumerate(time_series_splits):
            print(f"\nTraining on fold {fold_idx+1}/{len(time_series_splits)}")

            X_fold_train = X_with_idx.iloc[train_idx].copy()
            X_fold_test = X_with_idx.iloc[test_idx].copy()
            y_fold_train = y_with_idx.iloc[train_idx].copy()
            y_fold_test = y_with_idx.iloc[test_idx].copy()

            train_periods = X_fold_train['year_month'].unique()
            test_periods = X_fold_test['year_month'].unique()
            print(f"  Training periods: {min(train_periods)} to {max(train_periods)}")
            print(f"  Testing periods: {min(test_periods)} to {max(test_periods)}")
            print(f"  Training samples: {len(X_fold_train)}, Testing samples: {len(X_fold_test)}")

            preprocessor = create_preprocessor(
                feature_config['numeric_features_for_pipe'],
                feature_config['categorical_features_for_pipe']
            )

            for model_name, model in models.items():
                print(f"  Training {model_name}...")

                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])

                pipeline.fit(X_fold_train, y_fold_train)

                y_pred = pipeline.predict(X_fold_test)

                metrics = calculate_metrics(y_fold_test, y_pred)

                model_results[model_name]['fold_metrics'].append(metrics)

                with mlflow.start_run(run_name=f"{model_name}_fold{fold_idx+1}", nested=True):
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("fold", fold_idx+1)
                    mlflow.log_param("train_periods", f"{min(train_periods)}-{max(train_periods)}")
                    mlflow.log_param("test_periods", f"{min(test_periods)}-{max(test_periods)}")

                    if hasattr(model, 'get_params'):
                        for param_name, param_value in model.get_params().items():
                            mlflow.log_param(param_name, param_value)

                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"fold_{metric_name}", metric_value)

                    if fold_idx == len(time_series_splits) - 1:
                        plot_files = create_plots(
                            y_fold_test, y_pred,
                            f"Fold {fold_idx+1} (Holdout)",
                            model_name,
                            fold_idx+1
                        )

                        for plot_name, plot_file in plot_files.items():
                            mlflow.log_artifact(plot_file)

                print(f"    MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")

            baseline_metrics = model_results['DummyRegressor']['fold_metrics'][-1]

            for model_name in [m for m in models.keys() if m != 'DummyRegressor']:
                model_metrics = model_results[model_name]['fold_metrics'][-1]

                delta_mae = model_metrics['mae'] - baseline_metrics['mae']
                delta_rmse = model_metrics['rmse'] - baseline_metrics['rmse']

                print(f"    {model_name} vs Baseline: ΔMAE={delta_mae:.4f}, ΔRMSE={delta_rmse:.4f}")

                with mlflow.start_run(run_name=f"{model_name}_fold{fold_idx+1}", nested=True):
                    mlflow.log_metric("delta_mae", delta_mae)
                    mlflow.log_metric("delta_rmse", delta_rmse)

        for model_name, results in model_results.items():
            fold_metrics = results['fold_metrics']
            avg_metrics = {
                metric: np.mean([fold[metric] for fold in fold_metrics])
                for metric in fold_metrics[0].keys()
            }
            model_results[model_name]['avg_metrics'] = avg_metrics

            with mlflow.start_run(run_name=f"{model_name}_summary", nested=True):
                for metric_name, metric_value in avg_metrics.items():
                    mlflow.log_metric(f"avg_{metric_name}", metric_value)

        holdout_results = {
            model_name: results['fold_metrics'][-1]
            for model_name, results in model_results.items()
        }

        best_model_name = min(holdout_results.keys(), key=lambda k: holdout_results[k]['mae'])
        best_model_metrics = holdout_results[best_model_name]

        print(f"\nBest model on holdout set: {best_model_name}")
        print(f"  MAE: {best_model_metrics['mae']:.4f}")
        print(f"  RMSE: {best_model_metrics['rmse']:.4f}")
        print(f"  R²: {best_model_metrics['r2']:.4f}")
        print(f"  Spearman ρ: {best_model_metrics['spearman']:.4f}")

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_model_mae", best_model_metrics['mae'])
        mlflow.log_metric("best_model_rmse", best_model_metrics['rmse'])
        mlflow.log_metric("best_model_r2", best_model_metrics['r2'])
        mlflow.log_metric("best_model_spearman", best_model_metrics['spearman'])

        X_holdout_train = X_with_idx.iloc[holdout_train_idx].copy()
        y_holdout_train = y_with_idx.iloc[holdout_train_idx].copy()
        X_holdout_test = X_with_idx.iloc[holdout_test_idx].copy()
        y_holdout_test = y_with_idx.iloc[holdout_test_idx].copy()

        final_preprocessor = create_preprocessor(
            feature_config['numeric_features_for_pipe'],
            feature_config['categorical_features_for_pipe']
        )

        best_model = models[best_model_name]

        final_pipeline = Pipeline([
            ('preprocessor', final_preprocessor),
            ('model', best_model)
        ])

        final_pipeline.fit(X_holdout_train, y_holdout_train)

        saved_paths = persist_best_artifacts(
            best_model=final_pipeline['model'],
            final_preprocessor=final_pipeline['preprocessor'],
            final_feature_config=feature_config
        )

        for artifact_path in saved_paths.values():
            mlflow.log_artifact(artifact_path)

        return {
            'model_results': model_results,
            'holdout_results': holdout_results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'final_preprocessor': final_preprocessor,
            'final_feature_config': feature_config
        }


def main():
    print("Starting flight delay model training...")

    input_file = PROJECT_ROOT / 'data' / 'bts_processed' / 'flights_month_origin_carrier_base.parquet'

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading aggregated data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Data shape: {df.shape}")

    df = df.sort_values('year_month')

    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN, 'year_month']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    time_periods = df['year_month'].unique()
    print(f"Time periods in data: {min(time_periods)} to {max(time_periods)}")

    feature_config = {
        'order_of_columns_in_X_train': FEATURE_COLUMNS,
        'numeric_features_for_pipe': NUMERIC_FEATURES,
        'categorical_features_for_pipe': CATEGORICAL_FEATURES
    }

    print("\nFeature Configuration:")
    print(f"Total features: {len(FEATURE_COLUMNS)}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")

    X = df[FEATURE_COLUMNS + ['year_month']]
    y = df[TARGET_COLUMN]

    print("\nCreating time-based splits...")
    tss = TimeSeriesSplit(n_splits=5)

    time_period_index = pd.DataFrame({
        'year_month': time_periods,
        'period_idx': range(len(time_periods))
    })

    X_with_period_idx = X.merge(time_period_index, on='year_month', how='left')

    splits = []
    for train_idx, test_idx in tss.split(time_period_index['period_idx']):
        train_periods = time_period_index.iloc[train_idx]['year_month'].values
        test_periods = time_period_index.iloc[test_idx]['year_month'].values

        train_mask = X_with_period_idx['year_month'].isin(train_periods)
        test_mask = X_with_period_idx['year_month'].isin(test_periods)

        train_indices = X_with_period_idx[train_mask].index
        test_indices = X_with_period_idx[test_mask].index

        splits.append((train_indices, test_indices))

        print(f"Split: Train {min(train_periods)}-{max(train_periods)}, "
              f"Test {min(test_periods)}-{max(test_periods)}")

    results = train_and_evaluate_models(X, y, splits, feature_config)

    print("\nTraining completed successfully!")
    print(f"Best model: {results['best_model_name']}")
    print("Check MLflow UI for detailed experiment tracking.")


if __name__ == "__main__":
    main()