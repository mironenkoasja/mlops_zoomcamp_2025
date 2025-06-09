from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

PG_ENV = {
    'user': os.getenv("PG_USER"),
    'password': os.getenv("PG_PASSWORD"),
    'host': os.getenv("PG_HOST"),
    'port': os.getenv("PG_PORT"),
    'db_name': os.getenv("PG_DB_NAME"),
}

default_args = {
    "start_date": datetime(2023, 3, 1)
}

with DAG(
    dag_id='ml_pipeline_yellow_2023_03',
    schedule_interval=None,  # вручную запускать
    default_args=default_args,
    catchup=False
) as dag:

    download_ingest = BashOperator(
        task_id="download_and_ingest",
        bash_command=f"""
            python /opt/airflow/pipeline/download_ingest.py \
                --year 2023 --month 3 \
                --user {PG_ENV['user']} \
                --password {PG_ENV['password']} \
                --host {PG_ENV['host']} \
                --port {PG_ENV['port']} \
                --db_name {PG_ENV['db_name']} \
                --table_name raw_data_2023_03
        """
    )

    preprocess = BashOperator(
        task_id="preprocess_data",
        bash_command=f"""
            python /opt/airflow/pipeline/preprocess.py \
                --user {PG_ENV['user']} \
                --password {PG_ENV['password']} \
                --host {PG_ENV['host']} \
                --port {PG_ENV['port']} \
                --db_name {PG_ENV['db_name']} \
                --raw_table raw_data_2023_03 \
                --processed_table processed_data_2023_03
        """
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"""
            python /opt/airflow/pipeline/train_register.py \
                --user {PG_ENV['user']} \
                --password {PG_ENV['password']} \
                --host {PG_ENV['host']} \
                --port {PG_ENV['port']} \
                --db_name {PG_ENV['db_name']} \
                --table_name processed_data_2023_03
        """
    )

    download_ingest >> preprocess >> train_model
