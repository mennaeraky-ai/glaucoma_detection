# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# from utils.model_loader import 


# IMG_SIZE = (224, 224)

# st.title("🖼️ Test Glaucoma Detection Model")

# @st.cache_resource
# def load_model():
#     model_path = get_model_path()
#     return tf.keras.models.load_model(model_path)

# def preprocess_image(image: Image.Image):
#     image = image.convert("RGB").resize(IMG_SIZE)
#     arr = np.array(image) / 255.0
#     return np.expand_dims(arr, axis=0)

# model = load_model()

# uploaded_file = st.file_uploader(
#     "Upload a retinal fundus image",
#     type=["jpg", "jpeg", "png"]
# )

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, use_container_width=True)

#     if st.button("🔍 Predict"):
#         with st.spinner("Analyzing image..."):
#             x = preprocess_image(image)
#             prob = model.predict(x)[0][0]

#         label = "Glaucoma" if prob >= 0.5 else "Normal"
#         confidence = prob if prob >= 0.5 else 1 - prob

#         st.subheader("Prediction Result")
#         st.write(f"**Prediction:** `{label}`")
#         st.write(f"**Confidence:** `{confidence:.2%}`")

#         if label == "Glaucoma":
#             st.error("⚠️ Signs of glaucoma detected")
#         else:
#             st.success("✅ Normal fundus detected")

# st.caption("⚕️ For research purposes only – not a medical diagnosis.")
# pages/test_model.py

import streamlit as st
from utils.model_loader import load_model
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern

def app():
    st.title("🖼️ Test Glaucoma Detection Model")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        if st.button("🔍 Predict"):
            features = extract_features(image)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(features)[0][1]
            else:
                prob = model.predict(features)[0]

            label = "Glaucoma" if prob >= 0.5 else "Normal"
            confidence = prob if label == "Glaucoma" else 1 - prob

            st.subheader("Prediction Result")
            st.write(f"**Prediction:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2%}`")

            if label == "Glaucoma":
                st.error("⚠️ Signs of glaucoma detected")
            else:
                st.success("✅ Normal fundus detected")
