import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "models/LAST_glaucoma_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

st.title("🖼️ Test Glaucoma Detection Model")

model = load_model()

uploaded = st.file_uploader(
    "Upload Fundus Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):
            x = preprocess_image(image)
            prob = model.predict(x)[0][0]

        label = "Glaucoma" if prob >= 0.5 else "Normal"
        conf = prob if prob >= 0.5 else 1 - prob

        st.subheader("Result")
        st.write(f"**Prediction:** `{label}`")
        st.write(f"**Confidence:** `{conf:.2%}`")

        if label == "Glaucoma":
            st.error("⚠️ Glaucoma detected")
        else:
            st.success("✅ Normal fundus")
