import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
import json

with open('./onnx/model/resnet50-v1-12/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

def get_image(path, show=False):
    with Image.open(path) as img:
        img = np.array(img.convert('RGB'))
    if show:
        plt.imshow(img)
        plt.axis('off')
    return img

def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def predict(path):
    img = get_image(path, show=True)
    img = preprocess(img)
    ort_inputs = {session.get_inputs()[0].name: img}
    start_time = time.time() # Засекаем время перед предсказанием
    preds = session.run(None, ort_inputs)[0]
    end_time = time.time() # Засекаем время после предсказания
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    print('class=%s ; probability=%f' %(labels[a[0]], preds[a[0]]))
    return end_time - start_time # Возвращаем время выполнения предсказания

model_path = './onnx/model/resnet50-v1-12-int8/resnet50-v1-12-int8.onnx'
model = onnx.load(model_path)
session = ort.InferenceSession(model.SerializeToString())

img_path = './images/bird.jpeg'
prediction_times = []

# Запускаем 1000 предсказаний
for i in range(1000):
    time_taken = predict(img_path)
    prediction_times.append(time_taken)

# Сохраняем время предсказаний в формате массива и в файле JSON
with open('prediction_times.json', 'w') as f:
    json.dump(prediction_times, f)

print('Prediction times saved in prediction_times.json')
