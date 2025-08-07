# scripts/preprocess_images.py
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
DATA_PATH = "data"
CATEGORIES = ["real", "fake"]

def load_images():
    data = []
    labels = []
    for idx, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_PATH, category)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img)
                labels.append(idx)
            except:
                continue
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    X, y = load_images()
    X = X / 255.0  # normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy", y_test)


