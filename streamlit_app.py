import streamlit as st

st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔥 FORCE SIDEBAR RENDERING
st.sidebar.title("📌 Navigation")
st.sidebar.markdown(
    """
    Use the pages below to:
    - 🖼️ Test the trained model
    - 📊 Compare CNN architectures
    """
)

st.title("🧠 Glaucoma Detection System")
st.markdown(
    """
    **Deep Learning-based Retinal Fundus Analysis**

    This application allows:
    - Image-based glaucoma classification
    - Performance comparison of multiple CNN models
    """
)

st.info("👈 Use the sidebar to navigate between pages")
