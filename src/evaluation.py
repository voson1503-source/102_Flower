# evaluation.py
"""
Đánh giá mô hình ResNet50 đã huấn luyện trên dataset 102 Flowers.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
from model_training import IMG_SIZE, NUM_CLASSES

# ==============================
# ⚙️ Cấu hình đường dẫn
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_best.keras")

DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "processed")

# ==============================
# 🔹 Load model
# ==============================
def load_trained_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model không tồn tại tại {model_path}")
    model = load_model(model_path)
    print(f"✅ Model đã được load từ {model_path}")
    return model

# ==============================
# 🔹 Generator cho validation/test
# ==============================
from tensorflow.keras.applications.resnet50 import preprocess_input

def create_generator(subset="test", img_size=IMG_SIZE, batch_size=32):
    subset_dir = os.path.join(DATA_DIR, subset)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    gen = datagen.flow_from_directory(
        subset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False  # cần giữ nguyên thứ tự ảnh
    )
    return gen

# ==============================
# 🔹 Evaluation chung
# ==============================
def evaluate_model(model, gen, name="Validation"):
    loss, acc = model.evaluate(gen, verbose=1)
    print(f"✅ {name} Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return loss, acc

# ==============================
# 🔹 Accuracy riêng cho tập test
# ==============================
def test_accuracy(model, batch_size=32):
    """Tính accuracy chuẩn trên tập test"""
    test_gen = create_generator(subset="test", batch_size=batch_size)
    print(f"🧪 Đang đánh giá trên {len(test_gen.filenames)} ảnh test...")
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"✅ Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    return acc

# ==============================
# 🔹 Confusion Matrix
# ==============================
def plot_confusion_matrix(model, gen, class_names=None, title="Confusion Matrix"):
    y_true = gen.classes
    y_pred_probs = model.predict(gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, fmt="d", cmap='Blues')
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    print("📋 Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=[str(i) for i in range(NUM_CLASSES)],
        zero_division=0
    ))

# ==============================
# 📊 Vẽ biểu đồ Accuracy và Loss
# ==============================
def plot_training_curves(log_path):
    if not os.path.exists(log_path):
        print(f"⚠️ Không tìm thấy file log: {log_path}")
        return
    
    log_data = pd.read_csv(log_path)
    epochs = range(1, len(log_data) + 1)

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(epochs, log_data["accuracy"], label="Train Accuracy")
    plt.plot(epochs, log_data["val_accuracy"], label="Val Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(epochs, log_data["loss"], label="Train Loss")
    plt.plot(epochs, log_data["val_loss"], label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("✅ Biểu đồ đã được hiển thị.")

# ==============================
# 🔹 Top-k accuracy
# ==============================
def top_k_accuracy(model, gen, k=5, name="Validation"):
    y_true = gen.classes
    y_pred_probs = model.predict(gen, verbose=1)
    topk_acc = top_k_accuracy_score(y_true, y_pred_probs, k=k)
    print(f"✅ {name} Top-{k} Accuracy: {topk_acc:.4f}")
    return topk_acc

# ==============================
# 🔹 Main
# ==============================
if __name__ == "__main__":
    model = load_trained_model()

    # 🔹 Đánh giá trên tập test
    test_acc = test_accuracy(model)

    # 🔹 Vẽ confusion matrix và top-5 accuracy cho tập test
    test_gen = create_generator(subset="test")
    plot_confusion_matrix(model, test_gen, title="Test Confusion Matrix")
    top_k_accuracy(model, test_gen, k=5, name="Test")

    log_csv = os.path.join(MODEL_DIR, "training_log.csv")
    plot_training_curves(log_csv)

   