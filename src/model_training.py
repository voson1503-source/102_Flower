import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
)
from tensorflow.keras.applications.resnet50 import preprocess_input
import datetime

# ==============================
# ⚙️ Cấu hình chung
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 102
INITIAL_LR = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "train")
VAL_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "val")

# ==============================
# 🔹 Tạo dataset
# ==============================
def create_dataset(directory, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True, augment=False):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=shuffle
    )

    if augment:
        augment_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ])
        dataset = dataset.map(lambda x, y: (augment_layers(x, training=True), y))

    dataset = dataset.map(lambda x, y: (preprocess_input(x), y))
    return dataset.prefetch(tf.data.AUTOTUNE)

train_ds = create_dataset(TRAIN_DIR, augment=True)
train_ds = train_ds.repeat()
val_ds = create_dataset(VAL_DIR, augment=False, shuffle=False)

# ==============================
# 🔹 Xây dựng mô hình ResNet50 (Transfer Learning 2 giai đoạn)
# ==============================
def build_model(trainable_layers=50, learning_rate=INITIAL_LR):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze toàn bộ
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model, base_model

# ==============================
# 🔹 Huấn luyện mô hình
# ==============================
if __name__ == "__main__":
    model, base_model = build_model()
    model.summary()

    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "resnet50_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    earlystop_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )

    csv_logger_cb = CSVLogger(os.path.join(MODEL_DIR, "training_log.csv"))

    log_dir = os.path.join(MODEL_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Số bước huấn luyện chính xác
    num_train = sum([len(files) for _, _, files in os.walk(TRAIN_DIR)])
    num_val = sum([len(files) for _, _, files in os.walk(VAL_DIR)])
    steps_per_epoch = 32
    validation_steps = int(np.ceil(num_val / BATCH_SIZE))

    print(f"\n📊 Training images: {num_train}, Validation images: {num_val}")
    print(f"📈 Steps/epoch: {steps_per_epoch}, Val steps: {validation_steps}\n")

    # ==============================
    # Giai đoạn 1: Train phần head
    # ==============================
    print("🔹 Giai đoạn 1: Huấn luyện phần head...")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger_cb, tensorboard_cb]
    )

    # ==============================
    # Giai đoạn 2: Fine-tune ResNet50
    # ==============================
    print("\n🔹 Giai đoạn 2: Fine-tuning ResNet50 (50 lớp cuối)...")
    for layer in base_model.layers[-80:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger_cb, tensorboard_cb]
    )

    print("\n✅ Huấn luyện hoàn tất! Mô hình tốt nhất đã lưu tại:")
    print(os.path.join(MODEL_DIR, "resnet50_best.keras"))
