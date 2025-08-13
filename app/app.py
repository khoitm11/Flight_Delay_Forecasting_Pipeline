#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import joblib
import json
import argparse

PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_MODELS_DIRECTORY = os.path.join(PROJECT_ROOT_PATH, "saved_models")

MODEL_FILENAME_TO_LOAD = "final_best_flight_delay_model.joblib"
PREPROCESSOR_FILENAME_TO_LOAD = "data_preprocessor_for_best_model.joblib"
CONFIG_FILENAME_TO_LOAD = "preprocessor_feature_config_best_model.json"


def load_all_artifacts(models_storage_dir, model_filename, preprocessor_filename,
                       config_filename):
    model_full_path = os.path.join(models_storage_dir, model_filename)
    preprocessor_full_path = os.path.join(models_storage_dir, preprocessor_filename)
    config_full_path = os.path.join(models_storage_dir, config_filename)

    if not all(os.path.exists(p) for p in [model_full_path, preprocessor_full_path, config_full_path]):
        print("Error: One or more model/preprocessor/config files not found.")
        print("  Model path       :", model_full_path, "- Exists:", os.path.exists(model_full_path))
        print("  Preprocessor path:", preprocessor_full_path, "- Exists:", os.path.exists(preprocessor_full_path))
        print("  Config path      :", config_full_path, "- Exists:", os.path.exists(config_full_path))
        return None, None, None

    loaded_model = joblib.load(model_full_path)
    loaded_preprocessor = joblib.load(preprocessor_full_path)
    with open(config_full_path, 'r') as f_json_config:
        loaded_feature_config = json.load(f_json_config)
    print("Successfully loaded model, preprocessor, and feature config.")
    return loaded_model, loaded_preprocessor, loaded_feature_config


def prepare_dataframe_for_prediction(data_input_as_dict, expected_column_order, numeric_cols_list_for_na_fill):
    df_input_single_row = pd.DataFrame([data_input_as_dict]).reindex(columns=expected_column_order)

    for col_name_item in numeric_cols_list_for_na_fill:
        if col_name_item in df_input_single_row.columns:
            df_input_single_row[col_name_item] = pd.to_numeric(df_input_single_row[col_name_item],
                                                               errors='coerce').fillna(0.0)
    return df_input_single_row


def get_model_prediction(active_model, active_preprocessor, prepared_dataframe_input):
    processed_features_for_model = active_preprocessor.transform(prepared_dataframe_input)
    predicted_value = active_model.predict(processed_features_for_model)
    return predicted_value[0]


if __name__ == "__main__":
    model_input_features_ordered = [
        "airport", "arr_cancelled", "arr_diverted", "arr_flights", "carrier",
        "carrier_ct_rate", "late_aircraft_ct_rate", "month", "nas_ct_rate",
        "security_ct_rate", "weather_ct_rate"
    ]
    numeric_features_handled_by_pipe = [
        "arr_flights", "arr_cancelled", "arr_diverted", "carrier_ct_rate",
        "weather_ct_rate", "nas_ct_rate", "security_ct_rate", "late_aircraft_ct_rate"
    ]
    categorical_features_handled_by_pipe = ["month", "carrier", "airport"]

    parser = argparse.ArgumentParser(description="Predict flight delay rate.")

    for feature_name_arg in model_input_features_ordered:
        arg_type = int if feature_name_arg == "month" else (
            float if feature_name_arg in numeric_features_handled_by_pipe else str)
        is_arg_required = (
                feature_name_arg == "arr_flights" or feature_name_arg in categorical_features_handled_by_pipe)
        default_value_for_arg = 0.0 if feature_name_arg in numeric_features_handled_by_pipe and not is_arg_required else None
        help_str = "Value for " + feature_name_arg
        if default_value_for_arg is not None:
            parser.add_argument(("--" + feature_name_arg), type=arg_type, default=default_value_for_arg, help=help_str)
        else:
            parser.add_argument(("--" + feature_name_arg), type=arg_type, required=is_arg_required, help=help_str)

    parser.add_argument("--model_file", default=MODEL_FILENAME_TO_LOAD, help="Model filename (.joblib)")
    parser.add_argument("--preprocessor_file", default=PREPROCESSOR_FILENAME_TO_LOAD,
                        help="Preprocessor filename (.joblib)")
    parser.add_argument("--config_file", default=CONFIG_FILENAME_TO_LOAD, help="Feature config filename (.json)")

    parsed_args = parser.parse_args()

    ml_model_loaded, data_preprocessor_loaded, feature_config_loaded = load_all_artifacts(
        SAVED_MODELS_DIRECTORY, parsed_args.model_file, parsed_args.preprocessor_file, parsed_args.config_file
    )

    if ml_model_loaded and data_preprocessor_loaded and feature_config_loaded:
        col_order_from_loaded_config = feature_config_loaded.get('order_of_columns_in_X_train', [])
        numeric_cols_from_loaded_config = feature_config_loaded.get('numeric_features_for_pipe', [])

        if not col_order_from_loaded_config:
            print("Error: Config file missing 'order_of_columns_in_X_train'.")
        else:
            input_values_dict = {col: getattr(parsed_args, col, np.nan) for col in col_order_from_loaded_config}
            input_dataframe_ready = prepare_dataframe_for_prediction(
                input_values_dict,
                col_order_from_loaded_config,
                numeric_cols_from_loaded_config
            )

            has_critical_nan_in_numeric = False
            if numeric_cols_from_loaded_config:
                if input_dataframe_ready[numeric_cols_from_loaded_config].isnull().any().any():
                    has_critical_nan_in_numeric = True

            if not input_dataframe_ready.empty and not has_critical_nan_in_numeric:
                prediction = get_model_prediction(ml_model_loaded, data_preprocessor_loaded, input_dataframe_ready)

                if prediction is not None:
                    print("\n--- FLIGHT DELAY PREDICTION RESULT ---")
                    print("  Input Features:")
                    for feature_used in col_order_from_loaded_config:
                        print("    ", feature_used, ":", input_values_dict.get(feature_used))
                    print("  => Predicted flight delay rate:", round(prediction, 4))
            else:
                print("Error: Invalid input data or NaN values remain after preparation.")
                if has_critical_nan_in_numeric:
                    print("  NaN details in numeric columns:")
                    print(input_dataframe_ready[numeric_cols_from_loaded_config].isnull().sum())
    else:
        print("Exit due to failure loading model/preprocessor/config.")