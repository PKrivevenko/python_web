import os
from dotenv import load_dotenv

load_dotenv()

# Загрузка переменных окружения из .env файла
MODEL_DIR = os.getenv("MODEL_DIR", "/models")
MAX_MODELS = int(os.getenv("MAX_MODELS", 5))
N_JOBS = int(os.getenv("N_JOBS", 4))