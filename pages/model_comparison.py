import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title("📊 Model Performance Comparison")

    # ===============================
    # Pretraining Strategy Comparison
    # ===============================
    st.header("🧠 Pretraining Strategy Comparison")

    pretraining_data = {
        "Model": [
            "DenseNet121_CheXNet",
            "DenseNet121_Fundus",
            "ResNet50_Fundus"
        ],
        "Precision": [0.899038, 0.856846, 0.716904],
        "Recall":    [0.748,    0.826,    0.704],
        "F1_Score":  [0.816594, 0.841141, 0.710394],
        "Accuracy":  [0.870170, 0.879444, 0.778207]
    }

    df_pretrain = pd.DataFrame(pretraining_data)
    st.dataframe(df_pretrain, use_container_width=True)

    metric_pretrain = st.selectbox(
        "Select metric (Pretraining)",
        ["Accuracy", "Precision", "Recall", "F1_Score"]
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_pretrain["Model"], df_pretrain[metric_pretrain])
    ax1.set_ylabel(metric_pretrain)
    ax1.set_title(f"{metric_pretrain} – Pretraining Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig1)

    # ===============================
    # CNN Architecture Comparison
    # ===============================
    st.header("🏗️ CNN Architecture Comparison (Threshold = 0.5)")

    architecture_data = {
        "Model": [
            "ResNet50",
            "DenseNet121",
            "VGG16",
            "EfficientNetB1",
            "Xception",
            "MobileNet"
        ],
        "Accuracy":  [0.8717, 0.8717, 0.8601, 0.8570, 0.8447, 0.8192],
        "Precision": [0.8408, 0.9196, 0.8289, 0.8588, 0.8300, 0.7247],
        "Recall":    [0.824,  0.732,  0.804,  0.754,  0.752,  0.858],
        "F1":        [0.8323, 0.8151, 0.8162, 0.8030, 0.7891, 0.7857],
        "AUC":       [0.9327, 0.9326, 0.9290, 0.9258, 0.9181, 0.9122]
    }

    df_arch = pd.DataFrame(architecture_data)
    st.dataframe(df_arch, use_container_width=True)

    metric_arch = st.selectbox(
        "Select metric (Architecture)",
        ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(df_arch["Model"], df_arch[metric_arch])
    ax2.set_ylabel(metric_arch)
    ax2.set_title(f"{metric_arch} – CNN Architecture Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig2)

    st.markdown("""
    **Notes:**
    - Threshold fixed at **0.5**
    - DenseNet121 shows stable AUC
    - Fundus fine-tuning improves F1
    """)
