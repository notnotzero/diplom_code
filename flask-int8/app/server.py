from flask import Flask, request, jsonify
import requests
from PIL import Image
import numpy as np
import cv2
import onnx
import onnxruntime as ort
application = Flask(__name__)
import psutil

with open('../synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

# Load the ONNX model
model_path = '../resnet50-v1-12-int8.onnx'
model = onnx.load(model_path)
session = ort.InferenceSession(model.SerializeToString())

@application.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data['image_url']

    # Download image from URL
    img = Image.open(requests.get(url, stream=True).raw)
    img = np.array(img.convert('RGB'))

    # Preprocess image
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

    # Make prediction
    ort_inputs = {session.get_inputs()[0].name: img}
    preds = session.run(None, ort_inputs)[0]
    preds = np.squeeze(preds)
    a = np.argsort(preds)[::-1]
    process = psutil.Process()
    memory_info = process.memory_info()
    print( f"Memory usage: {memory_info.rss / (1024 ** 2):.2f} MB")
    result = {'class': labels[a[0]], 'probability': float(preds[a[0]])}
    return jsonify(result)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=50100)
