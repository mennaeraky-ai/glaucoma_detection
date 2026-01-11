import os
import gdown
import streamlit as st

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "LAST_glaucoma_model.keras")

GDRIVE_FILE_ID = "11btPBNR74na_NjjnjrrYT8RSf8ffiumo"

@st.cache_resource
def get_model_path():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)

        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        st.info("⬇️ Downloading model from Google Drive...")

        gdown.download(
            url=url,
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True   # 🔥 THIS FIXES THE ERROR
        )

    return MODEL_PATH
