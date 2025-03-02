# model_prediction_V2.py
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
MODEL_PATH = os.getenv('MODEL_PATH', 'models/model_v2.tf')

# Load the trained model outside the function
trained_model = load_model(MODEL_PATH)


def preprocess_image(img_content):
    nparr = np.frombuffer(img_content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@tf.function
def predict_condition(img):
    return trained_model(img)
