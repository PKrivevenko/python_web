from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import multiprocessing
from typing import List

app = FastAPI()

# Класс конфигурации модели
class ModelConfig(BaseModel):
    model_name: str
    max_models: int
    model_dir: str
    n_jobs: int  # количество ядер для обучения

# Поддерживаемые модели
models = {
    "logistic_regression": LogisticRegression,
    "svm": SVC,
    "decision_tree": DecisionTreeClassifier
}

# Кэш для загруженных моделей
loaded_models = {}

# Максимальное количество активных процессов
active_processes = 0
max_processes = 1  # Один процесс всегда будет зарезервирован для сервера

# Обработчик для загрузки конфигурации
def load_config():
    return {
        "model_dir": "/models",
        "max_models": 5,
        "n_jobs": 4
    }

# Создание нового процесса для обучения модели
def train_model(model_name, X, y, model_path):
    global active_processes
    if active_processes >= max_processes:
        raise HTTPException(status_code=400, detail="No available CPU cores for training.")
    active_processes += 1
    model = models[model_name]()
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    active_processes -= 1

@app.post("/fit/")
async def fit(model_config: ModelConfig, X: List[List[float]], y: List[int]):
    model_path = os.path.join(model_config.model_dir, f"{model_config.model_name}.pkl")
    if os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model already exists.")
    
    # Запускаем обучение в отдельном процессе
    multiprocessing.Process(target=train_model, args=(model_config.model_name, X, y, model_path)).start()
    return {"status": "Model training started."}

@app.post("/predict/")
async def predict(model_name: str, X: List[List[float]]):
    model_path = os.path.join(load_config()["model_dir"], f"{model_name}.pkl")
    if model_name not in loaded_models:
        try:
            with open(model_path, "rb") as f:
                loaded_models[model_name] = pickle.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found.")
    
    model = loaded_models[model_name]
    return {"predictions": model.predict(X).tolist()}

@app.post("/remove/")
async def remove(model_name: str):
    model_path = os.path.join(load_config()["model_dir"], f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")
    os.remove(model_path)
    if model_name in loaded_models:
        del loaded_models[model_name]
    return {"status": "Model removed."}

@app.post("/remove_all/")
async def remove_all():
    model_dir = load_config()["model_dir"]
    for model_file in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_file)
        os.remove(model_path)
    loaded_models.clear()
    return {"status": "All models removed."}
