const express = require("express");
const bodyParser = require("body-parser");
const ort = require("onnxruntime-node");

const app = express();
const port = 3000;

app.use(bodyParser.json());

// Load ONNX model
let session;
(async () => {
  try {
    session = await ort.InferenceSession.create("./resnet50-v1-12.onnx");
    console.log("ONNX model loaded successfully.");
  } catch (e) {
    console.error(`Failed to load ONNX model: ${e}`);
  }
})();

function createRandomTensor(shape) {
  const totalSize = shape.reduce((acc, val) => acc * val, 1);
  const data = new Float32Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    data[i] = 0.1; // Fill with random values between 0 and 1
  }
  return new ort.Tensor("float32", data, shape);
}

// Predict handler
app.post("/predict", async (req, res) => {
  try {
    const randomTensor = createRandomTensor([1, 3, 224, 224]);
    const results = await session.run({ data: randomTensor });
    const probabilities = results.resnetv17_dense0_fwd.cpuData;
    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    res.json({ predicted_class: maxIndex });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(port, () => {
  console.log(`Server is listening at http://localhost:${port}`);
});
