# ML Workflow Orchestration with Airflow

This project demonstrates a basic Machine Learning pipeline using **Apache Airflow** for orchestration. The pipeline includes the following stages:

1. **Data Loading** – Load processed NYC taxi data from a PostgreSQL table.
2. **Feature Engineering & Preprocessing** – Calculate duration and encode categorical features.
3. **Model Training, Evaluation & Model Logging** – Train a linear regression model using scikit-learn and evaluate it.

---

## Project Structure

├── dags/
│ └── ml_pipeline_dag.py # Airflow DAG definition
├── pipeline/
│ └── download_ingest.py 
  └── preprocess.py 
  └── train_register.py
├── docker-compose.yml # Services: Airflow, PostgreSQL, MLflow
├── mlruns/ # MLflow artifact store (mounted volume)
└── README.md

## Clone the repository and launch the project

```bash
docker-compose up --build
```
## Trigger the Airflow DAG

Once all services are up and running:

1. Open the Airflow UI at [http://localhost:8080](http://localhost:8080)
2. Log in with the default credentials:
   - **Username**: `airflow`
   - **Password**: `airflow`
3. Find the DAG named **`ml_pipeline_yellow_2023_03`**
4. Toggle it **on**
5. Click the **play** ▶️ button to trigger the pipeline


 