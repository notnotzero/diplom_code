# Используйте официальный образ Python как базовый
FROM arm64v8/python:3.8-slim
# Установите рабочую директорию в контейнере
WORKDIR /app

# Скопируйте файлы проекта в контейнер
COPY . /app

# Установите необходимые зависимости
RUN pip install -r requirements.txt


# Откройте порт 5001 для доступа к серверу
EXPOSE 5001
ENV FLASK_APP=server.py

# Запустите Flask-сервер
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
