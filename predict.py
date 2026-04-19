import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/resnet50_model.h5"
DATA_DIR = "data/images"
IMAGE_SIZE = (224, 224)

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded!")

# =========================
# RECREATE CLASS ORDER (IMPORTANT 🔥)
# =========================
datagen = ImageDataGenerator(rescale=1./255)

temp_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)

# Correct class mapping
class_indices = temp_gen.class_indices
class_mapping = {v: k for k, v in class_indices.items()}

print("\n📂 Classes:")
for k, v in class_mapping.items():
    print(k, "->", v)

# =========================
# PREPROCESS
# =========================
def preprocess(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICT
# =========================
def predict(image_path):

    img = preprocess(image_path)

    preds = model.predict(img)[0]

top_indices = preds.argsort()[-3:][::-1]

print("\n🔝 Top Predictions:")
for i in top_indices:
    print(f"{class_mapping[i]} → {preds[i]*100:.2f}%")

    print("\n========================")
    print("🐄 Breed:", class_mapping[idx])
    print("🔥 Confidence:", round(confidence, 2), "%")
    print("========================")

# =========================
# RUN
# =========================
image_path = input("\n📸 Enter image path: ")
predict(image_path)