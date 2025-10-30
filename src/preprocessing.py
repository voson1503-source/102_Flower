import os
import tarfile
import scipy.io
from PIL import Image
import shutil

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


def load_splits():
    """Đọc các chỉ số ảnh thuộc train/test/val từ setid.mat"""
    setid_path = os.path.join(DATA_DIR, "setid.mat")
    mat = scipy.io.loadmat(setid_path)
    train_ids = mat["trnid"][0] - 1  # trừ 1 để thành index Python
    val_ids = mat["valid"][0] - 1
    test_ids = mat["tstid"][0] - 1
    print(f"✅ Tập train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    return train_ids, val_ids, test_ids


def resize_and_save(image_path, save_path, size=(224, 224)):
    """Mở ảnh, resize và lưu lại"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    image.save(save_path, "JPEG", quality=95)


def preprocess_images(labels):
    """Chia dữ liệu theo setid.mat, resize 224x224 và lưu vào processed"""
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 🔹 Load phân chia dữ liệu chuẩn từ setid.mat
    train_ids, val_ids, test_ids = load_splits()

    image_dir = os.path.join(EXTRACTED_DIR, "jpg")
    image_files = sorted(os.listdir(image_dir))

    def save_subset(ids, subset):
        subset_dir = os.path.join(PROCESSED_DIR, subset)
        os.makedirs(subset_dir, exist_ok=True)
        for idx in ids:
            img_path = os.path.join(image_dir, image_files[idx])
            label = labels[idx]
            label_dir = os.path.join(subset_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            dest = os.path.join(label_dir, os.path.basename(img_path))
            resize_and_save(img_path, dest)
        print(f"✅ Đã xử lý {subset} ({len(ids)} ảnh).")

    save_subset(train_ids, "train")
    save_subset(val_ids, "val")
    save_subset(test_ids, "test")

    print("🎯 Hoàn tất chia dữ liệu & resize ảnh 224x224.")

def show_original_samples(num_images=5):
    """Hiển thị một vài ảnh gốc (trước khi xử lý)"""
    import random
    import matplotlib.pyplot as plt

    image_dir = os.path.join(EXTRACTED_DIR, "jpg")
    image_files = sorted(os.listdir(image_dir))
    sampled = random.sample(image_files, num_images)

    plt.figure(figsize=(15, 4))
    for i, fname in enumerate(sampled, 1):
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path)
        plt.subplot(1, num_images, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Ảnh gốc\n{fname}", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    extract_data()
    labels = load_labels()
    show_original_samples() 
    preprocess_images(labels)
