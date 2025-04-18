from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
import base64
import io
from PIL import Image

app = Flask(__name__)
model = joblib.load('model/svm_model.pkl')  # Load your trained model

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')  # grayscale
    image = image.resize((28, 28))  # match MNIST input
    image_array = np.array(image)
    image_array = image_array.reshape(1, -1)  # flatten for sklearn
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])  # remove base64 header
    input_image = preprocess_image(image_data)
    prediction = model.predict(input_image)[0]
    return jsonify({'digit': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
