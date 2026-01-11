import streamlit as st
import pages.test_model as test_model
import pages.model_comparison as model_comparison

# 1. Page Configuration
st.set_page_config(
    page_title="Glaucoma Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar Navigation UI
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Test Model", "Model Comparison"]
)

# 3. Logic to switch between pages
if page == "Home":
    st.title("🧠 Glaucoma Detection System")
    st.markdown("""
    ### Deep Learning-based Retinal Fundus Analysis
    Welcome to the Glaucoma Detection System. 
    
    **Available Modules:**
    1. **Test Model:** Upload a fundus image for real-time classification.
    2. **Model Comparison:** View performance metrics across different architectures.
    """)
    st.info("👈 Use the sidebar menu to select a module.")

elif page == "Test Model":
    # Run the function from 1_Test_Model.py
    test_model.app()  # Make sure you define `app()` in 1_Test_Model.py

elif page == "Model Comparison":
    # Run the function from 2_Model_Comparison.py
    model_comparison.app()  # Make sure you define `app()` in 2_Model_Comparison.py
