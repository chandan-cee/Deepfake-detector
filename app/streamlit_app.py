import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
model_path = os.path.join('model', 'deepfake_model.h5')
model = load_model(model_path)

# Constants
IMG_SIZE = 224  # ✅ Ensure this matches your training input sizec

# App UI
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🕵️‍♂️ Deepfake Detector")
st.markdown("Upload an image to determine if it's **Real** or **Fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((IMG_SIZE, IMG_SIZE))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = np.array(image) / 255.0  # normalize
    img = img.reshape(1, IMG_SIZE * IMG_SIZE * 3)  # flatten to (1, 150528)

    # Predict
    prediction = model.predict(img)[0][0]
 
    if prediction > 0.5:
        st.error(f"🧪 Prediction: **Fake** ({prediction:.2f})")
    else:
        st.success(f"✅ Prediction: **Real** ({1 - prediction:.2f})")
