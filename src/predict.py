import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_model.keras"

model = load_model(MODEL_PATH)


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image


def predict(image):

    image = preprocess_image(image)

    probability = model.predict(image)[0][0]

    label = "Fake" if probability > 0.5 else "Real"

    confidence = probability if label == "Fake" else 1 - probability

    return {
        "label": label,
        "confidence": float(confidence),
        "fake_probability": float(probability)
    }
