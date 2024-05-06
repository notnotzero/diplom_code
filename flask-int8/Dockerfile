FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0

WORKDIR /app
COPY ./app/ /app
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=app/server.py
CMD ["flask", "run", "--host", "0.0.0.0"]