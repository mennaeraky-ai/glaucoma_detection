import streamlit as st

st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"  # 🔥 FIX
)

st.title("🧠 Glaucoma Detection System")
st.markdown(
    """
    **Deep Learning-based Retinal Fundus Analysis**

    Use the sidebar to:
    - 🖼️ Test the trained model
    - 📊 Compare performance of multiple CNN architectures
    """
)

st.info("👈 Select a page from the sidebar to start")
