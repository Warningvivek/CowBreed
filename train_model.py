import tensorflow as tf
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# =========================
# CONFIG
# =========================
DATA_DIR = "data/images"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

num_classes = train_gen.num_classes

# =========================
# 🔵 MODEL 1: CNN
# =========================
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔵 Training CNN...\n")
cnn_history = cnn_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

cnn_model.save("models/cnn_model.h5")

# =========================
# 🔴 MODEL 2: MobileNetV2
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

mobilenet_model = Model(inputs=base_model.input, outputs=output)

mobilenet_model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🔴 Training MobileNetV2...\n")
mobilenet_history = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

mobilenet_model.save("models/final_model.keras")

# =========================
# 📊 COMPARISON
# =========================
cnn_val_acc = cnn_history.history['val_accuracy'][-1]
mobilenet_val_acc = mobilenet_history.history['val_accuracy'][-1]

cnn_train_acc = cnn_history.history['accuracy'][-1]
mobilenet_train_acc = mobilenet_history.history['accuracy'][-1]

print("\n📊 FINAL COMPARISON:")
print(f"CNN Train Accuracy: {cnn_train_acc:.4f}")
print(f"CNN Val Accuracy: {cnn_val_acc:.4f}")

print(f"MobileNet Train Accuracy: {mobilenet_train_acc:.4f}")
print(f"MobileNet Val Accuracy: {mobilenet_val_acc:.4f}")

# Winner
if mobilenet_val_acc > cnn_val_acc:
    winner = "MobileNetV2"
else:
    winner = "CNN"

print(f"\n🏆 BEST MODEL: {winner}")

# =========================
# SAVE COMPARISON JSON
# =========================
results = {
    "cnn": {
        "train_accuracy": float(cnn_train_acc),
        "val_accuracy": float(cnn_val_acc)
    },
    "mobilenet": {
        "train_accuracy": float(mobilenet_train_acc),
        "val_accuracy": float(mobilenet_val_acc)
    },
    "best_model": winner
}

with open("results/model_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("📁 Comparison saved!")

# =========================
# 📊 GRAPH
# =========================
plt.figure()

plt.plot(cnn_history.history['val_accuracy'], label='CNN Val', linestyle='--')
plt.plot(mobilenet_history.history['val_accuracy'], label='MobileNet Val')

plt.plot(cnn_history.history['accuracy'], label='CNN Train', linestyle=':')
plt.plot(mobilenet_history.history['accuracy'], label='MobileNet Train')

plt.title("CNN vs MobileNetV2 Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("results/comparison_graph.png")
plt.show()

print("📊 Graph saved!")