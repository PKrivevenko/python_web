# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем зависимости
RUN pip install fastapi uvicorn scikit-learn

# Копируем код в контейнер
COPY ./app /app
WORKDIR /app

# Указываем команду для старта FastAPI сервера
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]