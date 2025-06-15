# Homework Module 04: Deployment

## Run the prediction service

```bash
docker run \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data:/app/data \ 
  predict-service \
  --year 2023 \
  --month 5
```
