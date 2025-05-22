import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os


# Download model from Google Drive if not exists
model_path = "tomato_model.h5"
drive_file_id = "1b7zHOkpghNoKw_OkUZCiN6QEopAs89kV"  # Replace with your actual ID
gdown_url = f"https://drive.google.com/uc?id={drive_file_id}"

if not os.path.exists(model_path):
    st.info("Downloading model...")
    gdown.download(gdown_url, model_path, quiet=False)

model = load_model("tomato_model.h5")
class_names = ['Bacterial Spot', 'Late Blight', 'Leaf Mold', 'Early Blight', 'Healthy']  # Adjust as needed

st.title("🍅 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")

