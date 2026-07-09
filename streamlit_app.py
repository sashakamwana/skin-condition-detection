import streamlit as st
from fastai.vision.all import *
from PIL import Image
from pathlib import Path
import requests

st.title("Skin Condition Detection")
st.write("Upload a skin image and the model will predict the most likely class.")

st.warning(
    "This app is for learning/demo purposes only and is not a medical diagnosis tool."
)

MODEL_URL = "https://github.com/sashakamwana/skin-condition-detection/releases/download/v1.0/export.2.pkl"
MODEL_PATH = Path("export.pkl")

def download_model():
    if MODEL_PATH.exists():
        return

    with st.spinner("Downloading model..."):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_model():
    download_model()
    return load_learner(MODEL_PATH)

learn = load_model()
labels = learn.dls.vocab

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded image", width="stretch")

        st.write("Running prediction...")

        pred, pred_idx, probs = learn.predict(img)

        st.subheader("Prediction")
        st.write(f"**{pred}**")

        st.subheader("Probabilities")
        for i, label in enumerate(labels):
            st.write(f"{label}: {float(probs[i]):.4f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
