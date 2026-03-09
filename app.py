import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(name)
CORS(app)

model = tf.keras.models.load_model('mri_detector_model.h5')

def prepare_image(image, target_size=(224, 224)):
if image.mode != "RGB":
image = image.convert("RGB")
image = image.resize(target_size)
img_array = tf.keras.preprocessing.image.img_to_array(image)
img_array = np.expand_dims(img_array, axis=0)
return img_array / 255.0

@app.route("/", methods=["GET"])
def home():
return "NeuroVision API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
try:
if 'image' not in request.files:
return jsonify({"error": "No image uploaded"}), 400
file = request.files["image"]
image = Image.open(io.BytesIO(file.read()))
processed_image = prepare_image(image)
prediction = model.predict(processed_image)
prob = float(prediction[0][0])
result = "MRI" if prob > 0.5 else "Not_MRI"
return jsonify({"result": result, "confidence": round(prob if result == "MRI" else 1 - prob, 4)})
except Exception as e:
return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
port = int(os.environ.get("PORT", 5000))

app.run(host='0.0.0.0', port=port)
