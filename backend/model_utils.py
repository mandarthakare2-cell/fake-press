
# model_utils.py
import os
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')

def predict_image(pil_image):
    # Placeholder predict - in real use, load a trained model and run inference.
    # Here we just return a dummy response if model file doesn't exist.
    if not os.path.exists(MODEL_PATH):
        return {'label': 'unknown', 'probability': 0.0, 'note': 'No model.h5 found. Place your trained model at backend/model.h5'}
    # If model exists, user can replace this with actual loading/predict code.
    return {'label': 'real', 'probability': 0.75}
