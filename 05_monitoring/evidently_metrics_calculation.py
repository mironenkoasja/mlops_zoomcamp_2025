import datetime
import time
import logging
import random
import pandas as pd
import joblib
import psycopg

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnQuantileMetric,
    RegressionQualityMetric
)

# Логгирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
SEND_TIMEOUT = 10

# SQL-команда создания таблицы
create_table_statement = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics(
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT,
    fare_amount_quantile_50 FLOAT,
    regression_mae FLOAT
)
"""

# Данные и модель
reference_data = pd.read_parquet('data/reference.parquet')
raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')

reference_data['duration_min'] = (reference_data.lpep_dropoff_datetime - reference_data.lpep_pickup_datetime).dt.total_seconds() / 60
raw_data['duration_min'] = (raw_data.lpep_dropoff_datetime - raw_data.lpep_pickup_datetime).dt.total_seconds() / 60

with open('models/lin_reg.bin', 'rb') as f_in:
    model = joblib.load(f_in)

# Настройка колонок
begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target='duration_min'
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    RegressionQualityMetric(target="duration_min", prediction="prediction")
])

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if not res.fetchall():
            conn.execute("CREATE DATABASE test;")

    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
        conn.execute(create_table_statement)

def calculate_and_store_metrics(cursor, i):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
        (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ]
    
    if len(current_data) == 0:
         print(f"No data for day {i}, skipping.")

    current_data[num_features] = current_data[num_features].fillna(0) 
    current_data[cat_features] = current_data[cat_features].fillna('-1')  # или 'missing'
    current_data['prediction'] = model.predict(current_data[num_features + cat_features])
    reference_data[num_features] = reference_data[num_features].fillna(0)
    reference_data[cat_features] = reference_data[cat_features].fillna('-1')
    reference_data['prediction'] = model.predict(reference_data[num_features + cat_features])
    print("DEBUUUUUG")
    print(current_data[['duration_min', 'prediction']].isna().sum())
    
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
    #fare_amount_quantile = result['metrics'][3]['result']['current'].get('value', None)
    #regression_mae = None
    #try:
    #     regression_mae = result['metrics'][4]['result']['mean_absolute_error']
    #except (KeyError, TypeError):
    #     print("Could not extract regression_mae for day %s", i)

    cursor.execute(
        """
        INSERT INTO dummy_metrics(
            timestamp, prediction_drift, num_drifted_columns, share_missing_values,
            fare_amount_quantile_50, regression_mae
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (begin + datetime.timedelta(i), prediction_drift, num_drifted_columns,
         share_missing_values, fare_amount_quantile, regression_mae)
    )

def main():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)

    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        for i in range(0, 30):
            with conn.cursor() as cursor:
                calculate_and_store_metrics(cursor, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            last_send = new_send
            logging.info("data sent")

if __name__ == '__main__':
    main()
