#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from datetime import datetime


def main():
    print("Starting data preprocessing...")

    os.makedirs('data/bts_processed', exist_ok=True)

    input_file = 'data/bts_raw/bts_flight_delays_cancellations_2019_2023_sample3m.csv'
    output_file = 'data/bts_processed/flights_raw_clean.parquet'

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    dtypes = {
        'AIRLINE': 'str',
        'AIRLINE_CODE': 'str',
        'CARRIER': 'str',
        'ORIGIN': 'str',
        'DEST': 'str',
        'CANCELLED': 'float64',
        'DIVERTED': 'float64',
        'ARR_DELAY': 'float64',
        'DEP_DELAY': 'float64',
        'TAXI_IN': 'float64',
        'TAXI_OUT': 'float64',
        'DISTANCE': 'float64',
        'DELAY_DUE_CARRIER': 'float64',
        'DELAY_DUE_WEATHER': 'float64',
        'DELAY_DUE_NAS': 'float64',
        'DELAY_DUE_SECURITY': 'float64',
        'DELAY_DUE_LATE_AIRCRAFT': 'float64'
    }

    print(f"Reading CSV file: {input_file}")
    try:
        df = pd.read_csv(
            input_file,
            dtype=dtypes,
            parse_dates=['FL_DATE'],
            low_memory=False
        )
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

    print("Original columns:", df.columns.tolist())

    column_mapping = {}

    if 'AIRLINE_CODE' in df.columns:
        column_mapping['AIRLINE_CODE'] = 'carrier'
    elif 'CARRIER' in df.columns:
        column_mapping['CARRIER'] = 'carrier'

    if 'ORIGIN' in df.columns:
        column_mapping['ORIGIN'] = 'airport'

    df = df.rename(columns=column_mapping)

    print("Extracting date components...")
    df['year'] = df['FL_DATE'].dt.year
    df['month'] = df['FL_DATE'].dt.month
    df['year_month'] = df['year'] * 100 + df['month']

    print("Standardizing boolean columns...")
    for col in ['CANCELLED', 'DIVERTED']:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    print("Filtering records with valid airport and carrier...")
    valid_mask = (df['airport'].notna() & df['airport'].str.strip().ne('') &
                  df['carrier'].notna() & df['carrier'].str.strip().ne(''))
    df = df[valid_mask]

    delay_cols = [col for col in df.columns if col.startswith('DELAY_DUE_')]
    for col in delay_cols:
        df[col] = df[col].fillna(0)

    print(f"Saving processed data to: {output_file}")
    df.to_parquet(output_file, index=False)

    print("\nData Summary:")
    print(f"Final data shape: {df.shape}")
    print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"Cancellation rate: {df['CANCELLED'].mean():.2%}")
    print(f"Diversion rate: {df['DIVERTED'].mean():.2%}")

    print("\nColumn Names After Processing:")
    print(df.columns.tolist())

    print("\nSample Data (First 3 rows):")
    print(df.head(3))

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()