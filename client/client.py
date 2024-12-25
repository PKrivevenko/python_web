import requests

BASE_URL = "http://localhost:8000"

def train_model(model_name, X, y):
    data = {
        "model_name": model_name,
        "max_models": 5,
        "model_dir": "/models",
        "n_jobs": 4
    }
    response = requests.post(f"{BASE_URL}/fit/", json=data, params={"X": X, "y": y})
    print(response.json())

def predict(model_name, X):
    response = requests.post(f"{BASE_URL}/predict/", json={"model_name": model_name, "X": X})
    print(response.json())

def remove_model(model_name):
    response = requests.post(f"{BASE_URL}/remove/", json={"model_name": model_name})
    print(response.json())

def remove_all_models():
    response = requests.post(f"{BASE_URL}/remove_all/")
    print(response.json())
