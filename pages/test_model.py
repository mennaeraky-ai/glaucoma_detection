import streamlit as st
import numpy as np
from PIL import Image

from utils.model_loader import get_model_path

IMG_SIZE = (224, 224)

def app():
    st.title("🖼️ Test Glaucoma Detection Model")

    @st.cache_resource
    def load_model():
        # Lazy-import tensorflow so the UI can still load even if TF isn't installed.
        try:
            import tensorflow as tf  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "TensorFlow is not installed. Install it to enable predictions."
            ) from e

        model_path = get_model_path()
        return tf.keras.models.load_model(model_path)

    def preprocess_image(image: Image.Image):
        image = image.convert("RGB").resize(IMG_SIZE)
        arr = np.array(image) / 255.0
        return np.expand_dims(arr, axis=0)

    # Load model with friendly error handling (don't crash the whole app).
    try:
        model = load_model()
    except Exception as e:
        st.error("Model is not available yet, so predictions are disabled.")
        st.code(str(e))
        st.caption("Tip: set env var `GLAUCOMA_MODEL_PATH` to a local .keras file.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image..."):
                x = preprocess_image(image)
                y = model.predict(x)
                # Support common binary output shapes: (1,1) or (1,)
                prob = float(np.ravel(y)[0])

            label = "Glaucoma" if prob >= 0.5 else "Normal"
            confidence = prob if prob >= 0.5 else 1 - prob

            st.subheader("Prediction Result")
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2%}`")

            if label == "Glaucoma":
                st.error("⚠️ Signs of glaucoma detected")
            else:
                st.success("✅ Normal fundus detected")

    st.caption("⚕️ For research purposes only – not a medical diagnosis.")
