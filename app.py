from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

app = Flask(__name__)

IMAGE_SIZE = (224,224)

# LOAD MODELS
cnn_model = load_model("models/cnn_model.h5")
mobilenet_model = load_model("models/final_model.keras")

class_names = sorted(os.listdir("data/images"))

# DISEASES
disease_map = {
    "Lakhimi": ["Mastitis", "Foot and Mouth Disease", "Milk Fever"],
    "Sahiwal": ["Heat Stress", "Tick Fever", "Bovine Babesiosis"],
    "Umblachery": ["Skin Infection", "Parasitic Disease", "Ringworm"],
    "siri": ["Respiratory Infection", "Digestive Disorder", "Bloat"]
}

# 🔥 NEW: BREED INFO
breed_info = {
    "Lakhimi": {"origin": "India", "description": "High milk yield breed"},
    "Sahiwal": {"origin": "India/Pakistan", "description": "Heat tolerant dairy breed"},
    "Umblachery": {"origin": "Tamil Nadu", "description": "Strong draught breed"},
    "siri": {"origin": "Bhutan", "description": "Mountain breed"}
}

def preprocess(image):
    img = load_img(image, target_size=IMAGE_SIZE)
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = preprocess(filepath)

    preds_cnn = cnn_model.predict(img)[0]
    preds_resnet = mobilenet_model.predict(img)[0]

    idx_cnn = np.argmax(preds_cnn)
    idx_resnet = np.argmax(preds_resnet)

    conf_cnn = float(preds_cnn[idx_cnn] * 100)
    conf_resnet = float(preds_resnet[idx_resnet] * 100)

    # BEST MODEL
    if conf_resnet > conf_cnn:
        preds = preds_resnet
        final_conf = conf_resnet
    else:
        preds = preds_cnn
        final_conf = conf_cnn

    final_idx = np.argmax(preds)
    breed = class_names[final_idx].strip()

    # diseases
    diseases = disease_map.get(breed, ["No major disease found"])

    # origin + description
    info = breed_info.get(breed, {})
    origin = info.get("origin", "Unknown")
    description = info.get("description", "No description available")

    # 🔥 OTHER BREEDS (top 3)
    top3 = np.argsort(preds)[-3:][::-1]
    others = []
    for i in top3[1:]:
        others.append({
            "breed": class_names[i],
            "conf": float(round(float(preds[i]*100), 2))
})

    return jsonify({
        "breed": breed,
        "confidence": round(final_conf, 2),
        "origin": origin,
        "description": description,
        "diseases": diseases,
        "others": others
    })

if __name__ == "__main__":
    app.run(debug=True)