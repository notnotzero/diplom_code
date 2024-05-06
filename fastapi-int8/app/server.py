from fastapi import FastAPI, HTTPException
import requests
from PIL import Image
import numpy as np
import cv2
import onnx
import onnxruntime as ort
from pydantic import BaseModel

app = FastAPI()

# Модель данных для запроса
class ImageURL(BaseModel):
    image_url: str

with open('./synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

# Загрузка модели
model_path = './resnet50-v1-12-int8.onnx'
model = onnx.load(model_path)
# Создание сессии для инференса модели
session = ort.InferenceSession(model.SerializeToString())

@app.post('/predict')
async def predict(image_data: ImageURL):
    # Загрузка изображения по URL
    image_url = image_data.image_url
    img = Image.open(requests.get(image_url, stream=True).raw)
    img = np.array(img.convert('RGB'))

    # Препроцессинг изображения
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0: y0 + 224, x0: x0 + 224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Предсказание
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    result = {'class': labels[a[0]], 'probability': float(preds[a[0]])}
    return result

