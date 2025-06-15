FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and script
COPY model.bin .
COPY predict.py .

# Default command
ENTRYPOINT ["python", "predict.py"]

