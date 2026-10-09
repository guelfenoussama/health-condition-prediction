# 🏥 Health Condition Multi-Class Classification & Inference Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![Flask](https://img.shields.io/badge/API-Flask%20%2B%20Gunicorn-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

An end-to-end Machine Learning solution developed as part of the **DataTalks.Club Machine Learning Zoomcamp**. This project features a full exploratory data analysis, memory-optimized feature engineering pipeline, hyperparameter optimization, model ensembling, and a containerized Flask REST API ready for production deployment.

---

## 👨‍💻 Project Developer & Competition Context

* **Developer**: Osama — 17-year-old Machine Learning Engineer from Algeria  
* **Competition Performance**: This pipeline and engineering strategy placed in the **Top 4.2%** globally in an international competitive machine learning challenge.

> *"Handling 700,000 records requires striking a precise balance between computational efficiency and model capacity. This repository demonstrates how custom feature engineering and blended tree-based ensembles deliver competition-grade results under real-world runtime constraints."*

![Leaderboard Proof / Competition Ranking](C:\Users\ASUS\oussama3\health\health-condition-prediction\image\Predicting Student Health Risk _ Kaggle - Google Chrome 10_9_2026 9_07_34 PM.png)
*Figure 1: Proof of competition ranking (Top 4.2%). Replace placeholder link with your screenshot URL.*

---

## 📌 Problem Description & Dataset Profile

In healthcare informatics, predicting health status and risk categories accurately enables early intervention and optimal medical resource allocation. This project formulates a **multi-class classification problem** aimed at predicting patient health risk categories using biological, behavioral, and clinical metrics.

### Dataset Overview
* **Scale**: ~700,000 tabular records.
* **Target Variable**: Multi-class health status classification.
* **Key Indicators**: Sleep duration, heart rate, BMI, calorie expenditure, step count, exercise duration, water intake, diet type, stress level, sleep quality, physical activity level, smoking status, alcohol consumption, and gender.

Processing 700,000 rows with complex feature generation can quickly saturate memory and cause slow training loops. Specialized memory optimization and efficiency techniques were integrated directly into the engineering pipeline to maintain stability and speed.

---

## ⚡ High-Scale Efficiency & Feature Engineering Architecture

![System Architecture & Inference Workflow](https://via.placeholder.com/800x400.png?text=Insert+System+Architecture+or+Pipeline+Diagram+Here)
*Figure 2: End-to-end pipeline architecture from raw data payload to ensemble prediction.*

To achieve high predictive performance on a 700k dataset without exhausting memory, several core engineering techniques were built:

1. **Memory Precision Optimization**: Downcasted numerical columns to `float32` and `int32` precision across intermediate DataFrames, reducing peak memory usage by over **50%** during feature extraction.
2. **Automated Feature Engineering (`AutoTabularFeatures`)**:
   * **Row-Level Statistical Profiles**: Computed row-wise statistics (mean, std, min, max) across numeric indicators to capture global health variance per record.
   * **Pairwise Interaction Engine**: Created safe multiplicative interaction features and ratio features across strongly correlated numeric variables.
   * **Group Aggregation Profiling**: Calculated group-based mean, standard deviation, and offset differences by aggregating numeric columns over categorical behavioral drivers (e.g., stress level, activity level).
   * **Frequency & Ordinal Encodings**: Applied categorical frequency mapping and label encoding with robust missing-value handling (`__MISSING__`).
3. **Variance Threshold Filtering**: Automatically purged zero-variance and low-information features produced during generation using `VarianceThreshold`.
4. **Weighted Ensemble Architecture**: Built a custom `WeightedEnsemble` class blending probabilistic outputs from **LightGBM** and **XGBoost** with optimized soft-voting weights (0.45 XGBoost / 0.55 LightGBM).

---

## 📂 Repository Structure

```
.
├── oussi.ipynb        # Primary Jupyter Notebook (EDA, Feature Pipeline, Hyperparameter Tuning, Export)
├── predict.py         # Flask Web API service with Gunicorn compatibility and unpickling hooks
├── test.py            # Integration test script for sending JSON payloads to the API
├── Dockerfile         # Docker container configuration with OpenMP runtime dependencies
├── model.bin          # Serialized model artifacts (Ensemble model, Transformers, Encoders)
├── train.csv          # Training dataset (~700,000 rows)
├── test.csv           # Evaluation dataset
├── Pipfile            # Pipenv environment definition
└── Pipfile.lock       # Locked dependency versions
```

---

## 📓 Notebook Pipeline Overview (`oussi.ipynb`)

The training pipeline in `oussi.ipynb` follows a clear 9-stage structure:

1. **Libraries & Environment Setup**: Loading core Data Science, Boosting, and Evaluation frameworks.
2. **Exploratory Data Analysis (EDA)**: Inspecting 700k records, data types, missing value ratios, and feature distributions.
3. **Target Label Encoding & Stratified Split**: Preserving multi-class target balance using `StratifiedShuffleSplit`.
4. **Automated Feature Pipeline Construction**: Fitting and executing `AutoTabularFeatures` across splits.
5. **Ensemble Architecture Design**: Formulating the soft-voting blend between XGBoost and LightGBM.
6. **Hyperparameter Optimization**: Cross-validation experiments tuning boosting parameters (`learning_rate=0.025`, `max_depth`, `subsample`).
7. **Class Weighting & Ensemble Calibration**: Balancing class imbalance penalties and assigning optimal model blending weights.
8. **Validation & Performance Scoring**: Evaluating out-of-fold performance using `balanced_accuracy_score`.
9. **Full Training Pipeline & Artifact Serialization**: Training on the full 700k dataset and exporting pipeline artifacts to `model.bin`.

---

## 🚀 Deployment & Running the Service

### Prerequisites
* **Python**: Version `3.10`
* **Docker**: Installed and running on your system ([Download Docker](https://docs.docker.com/get-docker/))

---

### Method 1: Local Environment via Pipenv

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies using Pipenv**:
   ```bash
   pip install pipenv
   pipenv install
   pipenv shell
   ```

3. **Start the Flask service**:
   ```bash
   python predict.py
   ```
   *The service will start on `http://0.0.0.0:9696`.*

4. **Run prediction test (In a second terminal window)**:
   ```bash
   python test.py
   ```

---

### Method 2: Containerized Deployment via Docker (Recommended)

1. **Build the Docker container image**:
   ```bash
   docker build -t health-prediction-service .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -it --rm -p 9696:9696 health-prediction-service
   ```

3. **Test the Dockerized endpoint**:
   Run `test.py` from your host machine to verify predictions:
   ```bash
   python test.py
   ```

---

## 🧪 API Interface & Testing

### REST API Specification
* **Endpoint**: `http://localhost:9696/predict`
* **HTTP Method**: `POST`
* **Content-Type**: `application/json`

### Sample Request (`test.py`)
```json
{
  "id": 690088,
  "sleep_duration": 5.35,
  "heart_rate": 64.9,
  "bmi": 23.48,
  "calorie_expenditure": 2745.0,
  "step_count": 14167.0,
  "exercise_duration": 59.5,
  "water_intake": 1.86,
  "diet_type": "veg",
  "stress_level": "high",
  "sleep_quality": "poor",
  "physical_activity_level": "active",
  "smoking": "occasional",
  "alcohol": "no",
  "gender": "male"
}
```

### Sample Curl Request
```bash
curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{
       "sleep_duration": 5.35,
       "heart_rate": 64.9,
       "bmi": 23.48,
       "calorie_expenditure": 2745.0,
       "step_count": 14167.0,
       "exercise_duration": 59.5,
       "water_intake": 1.86,
       "diet_type": "veg",
       "stress_level": "high",
       "sleep_quality": "poor",
       "physical_activity_level": "active",
       "smoking": "occasional",
       "alcohol": "no",
       "gender": "male"
     }'
```

### Sample API Output
```json
{
  "prediction": "High Risk",
  "probability": 0.8742
}
```

---

## 📜 License
This project is open-source software licensed under the [MIT License](LICENSE).