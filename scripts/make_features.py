#!/usr/bin/env python3
# 2025-08-13 07:54:43 UTC - Doanduydong2
import os
import pandas as pd
import numpy as np


def main():
    print("Starting feature engineering...")

    input_file = 'data/bts_processed/flights_raw_clean.parquet'
    output_file = 'data/bts_processed/flights_month_origin_carrier_base.parquet'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Reading cleaned data from: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"Data shape: {df.shape}")

    print("Creating indicators...")
    df['arr_valid'] = (df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)
    df['is_del15'] = (df['ARR_DELAY'] >= 15) & df['arr_valid']

    required_cols = [
        'year_month', 'airport', 'carrier', 'arr_valid', 'is_del15',
        'CANCELLED', 'DIVERTED', 'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER',
        'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df['carrier_delay_ind'] = df['DELAY_DUE_CARRIER'] > 0
    df['weather_delay_ind'] = df['DELAY_DUE_WEATHER'] > 0
    df['nas_delay_ind'] = df['DELAY_DUE_NAS'] > 0
    df['security_delay_ind'] = df['DELAY_DUE_SECURITY'] > 0
    df['late_aircraft_delay_ind'] = df['DELAY_DUE_LATE_AIRCRAFT'] > 0

    print("\nDelay indicators sample counts:")
    print(f"Valid arrivals: {df['arr_valid'].sum()} / {len(df)}")
    print(f"Delayed flights (≥15 min): {df['is_del15'].sum()} / {df['arr_valid'].sum()}")

    print("\nAggregating data by (year_month, airport, carrier)...")
    agg_dict = {
        'arr_valid': 'sum',
        'CANCELLED': 'sum',
        'DIVERTED': 'sum',
        'is_del15': 'sum',
        'carrier_delay_ind': 'mean',
        'weather_delay_ind': 'mean',
        'nas_delay_ind': 'mean',
        'security_delay_ind': 'mean',
        'late_aircraft_delay_ind': 'mean',
        'month': 'first',
    }

    grouped_df = df.groupby(['year_month', 'airport', 'carrier']).agg(agg_dict).reset_index()

    grouped_df = grouped_df.rename(columns={
        'arr_valid': 'arr_flights',
        'CANCELLED': 'arr_cancelled',
        'DIVERTED': 'arr_diverted',
        'is_del15': 'arr_del15',
        'carrier_delay_ind': 'carrier_ct_rate',
        'weather_delay_ind': 'weather_ct_rate',
        'nas_delay_ind': 'nas_ct_rate',
        'security_delay_ind': 'security_ct_rate',
        'late_aircraft_delay_ind': 'late_aircraft_ct_rate'
    })

    print("Calculating delay rate (target variable)...")
    grouped_df['delay_rate'] = grouped_df['arr_del15'] / grouped_df['arr_flights']
    grouped_df['delay_rate'] = np.clip(grouped_df['delay_rate'], 0, 1)

    rate_columns = [
        'carrier_ct_rate', 'weather_ct_rate', 'nas_ct_rate',
        'security_ct_rate', 'late_aircraft_ct_rate', 'delay_rate'
    ]

    for col in rate_columns:
        grouped_df[col] = grouped_df[col].fillna(0)

    print("Filtering groups with at least 10 flights...")
    grouped_df = grouped_df[grouped_df['arr_flights'] >= 10]

    print(f"Saving aggregated features to: {output_file}")
    grouped_df.to_parquet(output_file, index=False)

    print("\nFeature Engineering Summary:")
    print(f"Original data shape: {df.shape}")
    print(f"Aggregated data shape: {grouped_df.shape}")
    print(f"Unique time periods: {grouped_df['year_month'].nunique()}")
    print(f"Unique airports: {grouped_df['airport'].nunique()}")
    print(f"Unique carriers: {grouped_df['carrier'].nunique()}")
    print(f"Average delay rate: {grouped_df['delay_rate'].mean():.4f}")

    print("\nDelay Rate Distribution:")
    print(grouped_df['delay_rate'].describe())

    print("\nSample Aggregated Data (First 3 rows):")
    print(grouped_df.head(3))

    print("\nFeature engineering completed successfully!")


if __name__ == "__main__":
    main()