from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("breast_cancer_mobilenet.h5")

IMG_SIZE = 160

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        result = "Malignant (Cancer Detected)"
    else:
        result = "Benign (No Cancer Detected)"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)