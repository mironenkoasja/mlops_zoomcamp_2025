import pandas as pd
from sqlalchemy import create_engine
import pyarrow.parquet as pq
from pyarrow import Table
import requests
import tempfile
import os

def download_and_ingest(year, month, user, password, host, port, db_name, table_name):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    print(f"Downloading file from {url}")

    # Подключение к БД
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')

    # Скачиваем файл локально по частям (stream=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp:
        tmp_path = tmp.name
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1 МБ чанки
                if chunk:  # фильтруем пустые чанки
                    tmp.write(chunk)

    print(f"Saved to temporary file: {tmp_path}")

    # Читаем файл по row group
    parquet_file = pq.ParquetFile(tmp_path)
    num_row_groups = parquet_file.num_row_groups
    rows_loaded = 0

    for rg in range(num_row_groups):
        row_group_reader = parquet_file.read_row_group(rg, use_threads=True)
        batch_reader = row_group_reader.to_batches(max_chunksize=10_000)

        for batch in batch_reader:
            df_chunk = Table.from_batches([batch]).to_pandas()[['tpep_dropoff_datetime', 'tpep_pickup_datetime', 'PULocationID', 'DOLocationID', 'trip_distance']]
            df_chunk.to_sql(name=table_name, con=engine, if_exists='append', index=False)
            rows_loaded += len(df_chunk)
            print(f"Loaded batch of {len(df_chunk)} rows, total: {rows_loaded}")

    os.remove(tmp_path)
    print("Temporary file deleted.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--db_name', required=True)
    parser.add_argument('--table_name', required=True)
    args = parser.parse_args()

    download_and_ingest(**vars(args))
