import streamlit as st
import gdown
import os

MODEL_URL = "https://drive.google.com/uc?id=1a6L-Yy0hb7X5PE4VMEX-N2usdvDzLLzs"
MODEL_PATH = "model.pkl"

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

model_file = download_model()

st.title("Web App Model AI")

st.success(f"Model berhasil dimuat: {model_file}")

# Load model sesuai framework
# contoh sklearn:
#
# import joblib
# model = joblib.load(model_file)
#
# prediksi = model.predict(...)
