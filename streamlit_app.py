import os
from pathlib import Path

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


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


MODEL_URL = "https://github.com/sashakamwana/skin-condition-detection/releases/download/v1.0/model.pth"
MODEL_PATH = Path("model.pth")
CLASSES_PATH = Path("classes.txt")


def download_model():
    if MODEL_PATH.exists():
        return

    with st.spinner("Downloading model... please wait."):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


@st.cache_resource
def load_model():
    download_model()

    classes = load_classes()
    num_classes = len(classes)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model, classes


try:
    model, classes = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Model failed to load.")
    st.exception(e)
    st.stop()


image_tfms = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


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
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Predict", key="predict_button"):
            with st.spinner("Running prediction..."):
                x = image_tfms(img).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)[0]

                pred_idx = int(torch.argmax(probs))
                pred_label = classes[pred_idx]

            st.subheader("Prediction")
            st.write(f"**{pred_label}**")

            st.subheader("Probabilities")
            for i, label in enumerate(classes):
                st.write(f"{label}: {float(probs[i]):.4f}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
  
