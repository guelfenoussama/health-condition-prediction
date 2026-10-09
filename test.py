import requests


# Target endpoint URL for the Docker local prediction service
url = "http://localhost:9696/predict"

# Sample patient payload matching test.csv feature format
sample_patient = {
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

try:
    # Send HTTP POST request with patient JSON data
    response = requests.post(url, json=sample_patient)
    result = response.json()
    # Display prediction result and confidence score
    print("--The result of the prediction--")
    print(f"prediction (Prediction): {result.get('prediction')}")
    print(f"probablity (Probability): {result.get('probability')}")
except Exception as e:
    # Handle connection errors or unexpected API responses
    print(f"error {e}")