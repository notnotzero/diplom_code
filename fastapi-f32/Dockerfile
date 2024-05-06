# Используем официальный образ Python 3.11
FROM python:3.11-slim

# Устанавливаем зависимости для FastAPI
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файлы вашего приложения в контейнер
COPY ./app/ /app

# Устанавливаем зависимости Python
RUN pip install -r requirements.txt

EXPOSE 8888

# Команда для запуска FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]
