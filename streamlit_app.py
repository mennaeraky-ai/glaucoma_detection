import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Glaucoma Detection System")
st.markdown("""
### Deep Learning-based Retinal Fundus Analysis
Welcome to the Glaucoma Detection System.

**Available Modules (use the left sidebar):**
1. **Test Model:** Upload a fundus image for real-time classification.
2. **Model Comparison:** View performance metrics across different architectures.
""")
st.info("👈 Use the sidebar pages list to open **Test model** or **Model comparison**.")
