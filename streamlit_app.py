import streamlit as st
import pages.test_model as test_model
import pages.model_comparison as model_comparison

st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Test Model", "Model Comparison"]
)

if page == "Home":
    st.title("🧠 Glaucoma Detection System")
    st.markdown("""
    ### Deep Learning-based Retinal Fundus Analysis

    **Available Modules:**
    - Test Model
    - Model Comparison
    """)
    st.info("👈 Use the sidebar to navigate")

elif page == "Test Model":
    test_model.app()

elif page == "Model Comparison":
    model_comparison.app()
