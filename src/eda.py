"""
eda.py – Phân tích dữ liệu khám phá cho tập Oxford 102 Flowers.
Hiển thị ảnh kèm tên hoa thay vì số lớp.
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# Đường dẫn thư mục dữ liệu
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
DATA_DIR = os.path.abspath(DATA_DIR)
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FLOWER_NAMES_PATH = os.path.join(DATA_DIR, "flowernames.txt")


def load_flower_names():
    """Đọc danh sách tên hoa từ file flowernames.txt"""
    with open(FLOWER_NAMES_PATH, "r") as f:
        names = [line.strip() for line in f.readlines()]
    return names


def get_class_distribution(subset="train"):
    """Đếm số lượng ảnh trong mỗi lớp"""
    subset_dir = os.path.join(PROCESSED_DIR, subset)
    class_counts = {}
    for cls in os.listdir(subset_dir):
        cls_dir = os.path.join(subset_dir, cls)
        count = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")])
        class_counts[int(cls)] = count
    return class_counts


def plot_class_distribution(subset="train", top_n=20):
    """Vẽ biểu đồ phân bố lớp"""
    counts = get_class_distribution(subset)
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n])

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_counts.keys(), sorted_counts.values(), color='skyblue')
    plt.xlabel("Class ID")
    plt.ylabel("Số lượng ảnh")
    plt.title(f"Phân bố {top_n} lớp có nhiều ảnh nhất ({subset})")
    plt.tight_layout()
    plt.show()


def show_random_samples(subset="train", n_images=10):
    """Hiển thị ngẫu nhiên n ảnh trong tập train/test với tên hoa"""
    flower_names = load_flower_names()
    subset_dir = os.path.join(PROCESSED_DIR, subset)

    all_paths = []
    for cls in os.listdir(subset_dir):
        cls_dir = os.path.join(subset_dir, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith(".jpg")]
        all_paths.extend([(img, int(cls)) for img in imgs])

    sampled = random.sample(all_paths, min(n_images, len(all_paths)))

    plt.figure(figsize=(15, 6))
    for i, (img_path, label) in enumerate(sampled, 1):
        plt.subplot(2, 5, i)
        plt.imshow(Image.open(img_path))
        plt.axis("off")
        flower_name = flower_names[label - 1].title()  # lấy tên hoa
        plt.title(f"{flower_name}\n(ID: {label})", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_class_distribution("train")
    show_random_samples("train", 10)
