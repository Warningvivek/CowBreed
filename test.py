import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

DATA_DIR = "data/images"
IMAGE_SIZE = (224,224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical",
    shuffle=False   # 🔥 VERY IMPORTANT
)

model = load_model("models/model.h5")

# Predictions
preds = model.predict(val_gen)

y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

class_labels = list(val_gen.class_indices.keys())

print("\n✅ Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))