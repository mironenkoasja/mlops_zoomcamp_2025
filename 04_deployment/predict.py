#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    return parser.parse_args()


def read_data(filename):
    # Load and preprocess the dataset
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)

    # Calculate duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    # Filter out trips with unrealistic duration
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # Convert categorical columns to string
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def main(year, month):
    # File paths
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/predict_yellow_tripdata_{year:04d}-{month:02d}.parquet'

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Load model and vectorizer
    print('Loading model...')
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Load and preprocess data
    print(f'Loading data from {input_file}...')
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Prepare features and make predictions
    dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict(X)

    # Compute metrics
    mean_pred = y_pred.mean()
    mean_actual = df['duration'].mean()
    mae = abs(df['duration'] - y_pred).mean()

    print(f'Mean predicted duration: {mean_pred:.2f} minutes')
    print(f'Mean actual duration: {mean_actual:.2f} minutes')
    print(f'Mean absolute error (MAE): {mae:.2f} minutes')

    # Save predictions
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'prediction': y_pred
    })

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        index=False
    )

    print(f'Predictions saved to {output_file}')

    
    metrics = {
        "year": year,
        "month": month,
        "n_predictions": len(y_pred),
        "mean_prediction": round(mean_pred, 2),
        "mean_actual": round(mean_actual, 2),
        "mae": round(mae, 2),
        "std_prediction": round(y_pred.std(), 2)
    }

    metrics_file = f'output/metrics_{year:04d}_{month:02d}.json'
    with open(metrics_file, 'w') as f_out:
        json.dump(metrics, f_out, indent=4)

    print(f'Metrics saved to {metrics_file}')


if __name__ == '__main__':
    args = parse_args()
    main(args.year, args.month)
