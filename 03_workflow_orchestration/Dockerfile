FROM apache/airflow:2.10.5

ENV AIRFLOW_HOME=/opt/airflow

USER airflow
#RUN apt-get update -qq && apt-get install vim -qqq
# git gcc g++ -qqq

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pandas sqlalchemy psycopg2-binary
