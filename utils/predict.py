# utils/predict.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

def predict_image(image_path):
    model = load_model("model/deepfake_detector_model.h5")
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)[0][0]
    return "Fake" if prediction > 0.5 else "Real"

if __name__ == "__main__":
    image_path = input("Enter image path to classify: ")
    result = predict_image(image_path)
    print(f"The image is predicted to be: {result}")
