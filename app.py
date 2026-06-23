import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ==========================
# Download model dari Google Drive
# ==========================
FILE_ID = "1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"
MODEL_PATH = "model_buah_cnn.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    with st.spinner("Mengunduh model..."):
        gdown.download(url, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# GANTI sesuai urutan class saat training
# ==========================
class_names = [
    "Apel",
    "Pisang",
    "Jeruk",
    "Mangga",
    "Semangka"
]

IMG_SIZE = (224, 224)

st.title("🍎 Tebak Gambar Buah")

uploaded = st.file_uploader(
    "Upload gambar buah",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    kelas = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediksi : {class_names[kelas]}")
    st.write(f"Confidence : {confidence*100:.2f}%")
