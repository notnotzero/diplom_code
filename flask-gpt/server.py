from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
app = Flask(__name__)
cors = CORS(app, resources={r"/generate": {"origins": "*"}, r"/":{"origins": "*"} })


# Список доступных моделей
model_names = ['gpt2', 'gpt2-medium']
models = {}
tokenizers = {}

def load_model(model_name, model_dir='app/models'):
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print(f"Downloading {model_name} model to {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"Model {model_name} downloaded and saved to {model_path}")
    else:
        print(f"Loading {model_name} model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

# Предварительная загрузка моделей и токенизаторов
for model_name in model_names:
    models[model_name], tokenizers[model_name] = load_model(model_name)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    text = data.get('text', '')
    max_length = int(data.get('max_length', 50))
    model_name = data.get('model', 'gpt2')

    tokenizer = tokenizers[model_name]
    model = models[model_name]

    input_ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    print('starting server on 5001')
    app.run(port=5001)
