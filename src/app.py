# src/app.py
"""
Ứng dụng web dự đoán hoa sử dụng mô hình ResNet50 đã huấn luyện.
- Upload ảnh
- Nhấn nút "Dự đoán" để xem kết quả
- Hiển thị tên hoa, độ chính xác, biểu đồ xác suất
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ==============================
# ⚙️ Cấu hình
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "resnet50_best.keras")
FLOWER_NAMES_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "flowernames.txt")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "saved_models", "class_indices.npy")
IMG_SIZE = (224, 224)

# ==============================
# 🔹 Load model và class names
# ==============================
@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)

@st.cache_data
def load_flower_names():
    with open(FLOWER_NAMES_PATH, "r") as f:
        names = [line.strip() for line in f.readlines()]
    return names

model = load_trained_model()
flower_names = load_flower_names()
class_indices = np.load(CLASS_INDICES_PATH, allow_pickle=True).item()
idx_to_class = {v: k for k, v in class_indices.items()}

# ==============================
# 🔹 Tiền xử lý ảnh đầu vào
# ==============================
def preprocess_uploaded_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img

# ==============================
# 🌸 Giao diện chính
# ==============================
st.set_page_config(page_title="🌼 Flower Classifier", layout="centered")
st.title("🌸 Dự đoán loại hoa bằng ResNet50")
st.markdown("Tải lên ảnh hoa rồi nhấn **Dự đoán** để xem kết quả 🌺")

uploaded_file = st.file_uploader("Tải lên ảnh (.jpg hoặc .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Ảnh đã tải lên", use_column_width=True)
    
    # Nút bấm dự đoán
    if st.button("🔍 Dự đoán"):
        with st.spinner("⏳ Đang dự đoán..."):
            img_array, img = preprocess_uploaded_image(uploaded_file)
            preds = model.predict(img_array)
            top5_idx = np.argsort(preds[0])[-5:][::-1]
            top5_probs = preds[0][top5_idx]
            top5_labels = [flower_names[int(idx_to_class[i]) - 1] for i in top5_idx]

        # Hiển thị kết quả
        predicted_class = top5_labels[0]
        confidence = top5_probs[0] * 100
        st.success(f"🌼 **Loài hoa dự đoán:** {predicted_class}")
        st.info(f"🎯 **Độ chính xác:** {confidence:.2f}%")

        # Biểu đồ top-5
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top5_labels[::-1], top5_probs[::-1], color='skyblue')
        ax.set_xlabel("Xác suất dự đoán")
        ax.set_title("Top-5 dự đoán gần đúng nhất")
        st.pyplot(fig)
else:
    st.warning("👆 Hãy tải lên một ảnh hoa để bắt đầu.")

# ==============================
# 📊 Thông tin mô hình
# ==============================
st.sidebar.header("ℹ️ Thông tin mô hình")
st.sidebar.markdown("""
**Kiến trúc:** ResNet50  
**Dataset:** Oxford 102 Flowers  
**Số lớp:** 102  
**Framework:** TensorFlow + Keras  
""")
