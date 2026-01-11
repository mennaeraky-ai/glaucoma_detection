# import os
# import gdown
# import streamlit as st

# MODEL_DIR = "models"
# MODEL_PATH = os.path.join(MODEL_DIR, "LAST_glaucoma_model.keras")

# GDRIVE_FILE_ID = "11btPBNR74na_NjjnjrrYT8RSf8ffiumo"

# @st.cache_resource
# def get_model_path():
#     if not os.path.exists(MODEL_PATH):
#         os.makedirs(MODEL_DIR, exist_ok=True)

#         url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
#         st.info("⬇️ Downloading model from Google Drive...")

#         gdown.download(
#             url=url,
#             output=MODEL_PATH,
#             quiet=False,
#             fuzzy=True   # 🔥 THIS FIXES THE ERROR
#         )

#     return MODEL_PATH
import os
import gdown
import streamlit as st
import pickle

MODEL_DIR = "models"
MODEL_NAME = "LAST_glaucoma_model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

GDRIVE_FILE_ID = "11btPBNR74na_NjjnjrrYT8RSf8ffiumo"
MIN_MODEL_SIZE_MB = 1  # sklearn models are smaller

@st.cache_resource(show_spinner=False)
def load_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

        gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found.")
        st.stop()

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < MIN_MODEL_SIZE_MB:
        st.error("❌ Invalid or corrupted model file.")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model
