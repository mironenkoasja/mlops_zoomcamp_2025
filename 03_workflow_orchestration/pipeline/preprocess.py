import argparse
import pandas as pd
from sqlalchemy import create_engine

def preprocess_data(user, password, host, port, db_name, raw_table, processed_table):
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    df = pd.read_sql(f"SELECT * FROM {raw_table}", con=engine)

    df['duration'] = pd.to_datetime(df.tpep_dropoff_datetime) - pd.to_datetime(df.tpep_pickup_datetime)
    df['duration'] = df['duration'].dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df = df[['PULocationID', 'DOLocationID', 'trip_distance', 'duration']]

    df.to_sql(processed_table, con=engine, if_exists='replace', index=False)
    print(f"Processed and saved {len(df)} rows into table '{processed_table}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--db_name")
    parser.add_argument("--raw_table")
    parser.add_argument("--processed_table")
    args = parser.parse_args()

    preprocess_data(**vars(args))
