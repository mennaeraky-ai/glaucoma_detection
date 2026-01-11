import streamlit as st

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
    # This runs the code from your 1_Test_Model.py file
    st.title("🖼️ Test the Trained Model")
    st.write("Upload a retinal image to detect signs of Glaucoma.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        st.success("Image uploaded! Ready for processing...")

elif page == "Model Comparison":
    # This runs the code from your 2_Model_Comparison.py file
    st.title("📊 Compare CNN Architectures")
    st.write("Analysis of model performance across different metrics.")
    
    # Example table
    st.table({
        "Model": ["VGG16", "ResNet50", "InceptionV3"],
        "Accuracy": ["92%", "95%", "94%"],
        "F1-Score": [0.89, 0.93, 0.91]
    })
