import os

for folder in ["train", "test"]:
    base = folder
    for name in os.listdir(base):
        old_path = os.path.join(base, name)
        if os.path.isdir(old_path):
            new_name = f"{int(name):03d}"  # chuyển 1 → 001
            new_path = os.path.join(base, new_name)
            os.rename(old_path, new_path)
            print(f"✅ {old_path} → {new_path}")
