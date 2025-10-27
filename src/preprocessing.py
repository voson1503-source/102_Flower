import os
import tarfile
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
DATA_DIR = os.path.abspath(DATA_DIR)
EXTRACTED_DIR = os.path.join(DATA_DIR, "flowers")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FLOWER_NAMES_PATH = os.path.join(DATA_DIR, "flowernames.txt")


def extract_data():
    """Giải nén dữ liệu .tgz nếu chưa có"""
    tgz_path = os.path.join(DATA_DIR, "102flowers.tgz")
    if not os.path.exists(EXTRACTED_DIR):
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(path=EXTRACTED_DIR)
        print("✅ Giải nén dữ liệu thành công.")
    else:
        print("✅ Dữ liệu đã được giải nén.")


def load_labels():
    """Đọc file nhãn imagelabels.mat"""
    labels_path = os.path.join(DATA_DIR, "imagelabels.mat")
    mat = scipy.io.loadmat(labels_path)
    labels = mat["labels"][0]
    print(f"✅ Đã tải {len(labels)} nhãn hoa.")
    return labels


def preprocess_images(labels):
    """Chia dữ liệu train/test và lưu vào thư mục processed"""
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    image_dir = os.path.join(EXTRACTED_DIR, "jpg")
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def save_subset(paths, labels, subset):
        subset_dir = os.path.join(PROCESSED_DIR, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for img_path, label in zip(paths, labels):
            label_dir = os.path.join(subset_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            dest = os.path.join(label_dir, os.path.basename(img_path))
            shutil.copy(img_path, dest)

    save_subset(train_paths, y_train, "train")
    save_subset(test_paths, y_test, "test")
    print("✅ Đã tạo tập train/test và lưu trong thư mục processed/")


def show_sample_images(subset="train", num_images=10, img_size=(224, 224)):
    """Hiển thị ngẫu nhiên một số ảnh mẫu (mặc định 10 ảnh) kèm tên loài hoa."""
    names_map = {}
    if os.path.exists(FLOWER_NAMES_PATH):
        with open(FLOWER_NAMES_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                names_map[i] = line.strip()

    subset_dir = os.path.join(PROCESSED_DIR, subset)

    # 🔹 Lấy tất cả ảnh từ các lớp con
    all_images = []
    for cls in os.listdir(subset_dir):
        cls_dir = os.path.join(subset_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".jpg")]
        all_images.extend([(img, int(cls)) for img in imgs])

    # 🔹 Lấy ngẫu nhiên num_images ảnh
    sampled = random.sample(all_images, min(num_images, len(all_images)))

    cols = 5  # số ảnh mỗi hàng
    rows = (len(sampled) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))

    for i, (img_path, label) in enumerate(sampled, 1):
        image = Image.open(img_path).resize(img_size)
        label_name = names_map.get(label, f"Class {label}")
        plt.subplot(rows, cols, i)
        plt.imshow(image)
        plt.axis("off")
        plt.title(label_name, fontsize=9, pad=8)

    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    extract_data()
    labels = load_labels()
    preprocess_images(labels)
    print("📸 Hiển thị một vài ảnh mẫu sau khi xử lý...")
    show_sample_images(subset="train", num_images=10)
