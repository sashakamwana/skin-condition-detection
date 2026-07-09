import streamlit as st
from fastai.vision.all import *
from PIL import Image

st.title("Skin Condition Detection")
st.write("Upload a skin image and the model will predict the most likely class.")

@st.cache_resource
def load_model():
    return load_learner("export.pkl")

learn = load_model()
labels = learn.dls.vocab

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    pred, pred_idx, probs = learn.predict(img)

    st.subheader("Prediction")
    st.write(f"**{pred}**")

    st.subheader("Probabilities")
    for i, label in enumerate(labels):
        st.write(f"{label}: {float(probs[i]):.4f}")
