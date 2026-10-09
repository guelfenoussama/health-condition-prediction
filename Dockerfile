# Use official lightweight Python base image
FROM python:3.10-slim
# Set working directory inside container
WORKDIR /app
# Install OpenMP library required for LightGBM/XGBoost C++ backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Install required Python dependencies
RUN pip install --no-cache-dir \
    flask \
    gunicorn \
    scikit-learn \
    xgboost \
    lightgbm \
    pandas \
    numpy
# Copy trained model file and web service code into container    
COPY ["model.bin", "predict.py", "./"]
# Expose web service port
EXPOSE 9696
# Define command to run Flask app via Gunicorn server
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]