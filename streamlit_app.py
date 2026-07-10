import os
import requests
import streamlit as st
import torch
from fastai.vision.all import *
from PIL import Image
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(1)


st.set_page_config(
    page_title="Skin Condition Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("Skin Condition Detection")
st.write("Upload a skin image and the model will predict the most likely class.")

st.warning(
    "This app is for learning/demo purposes only and is not a medical diagnosis tool."
)


# This is the GitHub Release download link for your model file.
# Your release file is called export.2.pkl, but we save it locally as export.pkl.
MODEL_URL = "https://github.com/sashakamwana/skin-condition-detection/releases/download/v1.0/export.2.pkl"
MODEL_PATH = Path("export.pkl")


def download_model():
    """Download the model from GitHub Releases if it is not already downloaded."""
    if MODEL_PATH.exists():
        return

    with st.spinner("Downloading model... please wait."):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


@st.cache_resource
def load_model():
    """Load the exported fastai model on CPU."""
    download_model()
    return load_learner(MODEL_PATH, cpu=True)
try:
    learn = load_model()
    labels = learn.dls.vocab
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()


uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    key="skin_image_uploader"
)


if uploaded_file is None:
    st.info("Please upload an image to get a prediction.")

else:
    st.success("Image uploaded successfully.")

    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.write("Image opened successfully.")

        # This displays the uploaded image.
        # If this causes frontend issues again, comment out this line.
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", key="predict_button"):
            with st.spinner("Running prediction..."):
                fastai_img = PILImage.create(img)
                pred, pred_idx, probs = learn.predict(fastai_img)
                

            st.subheader("Prediction")
            st.write(f"**{pred}**")

            st.subheader("Probabilities")
            for i, label in enumerate(labels):
                st.write(f"{label}: {float(probs[i]):.4f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
