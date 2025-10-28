# model_training.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.applications.resnet50 import preprocess_input
import datetime

# ==============================
# ⚙️ Cấu hình chung
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 102
LEARNING_RATE = 1e-4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "train")
VAL_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "processed", "test")


# ==============================
# 🔹 Tạo dataset từ thư mục
# ==============================
def create_dataset(directory, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True, augment=False):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='int',
        shuffle=shuffle
    )

    # Augmentation chỉ áp dụng cho training
    if augment:
        data_augment = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1,0.1)
        ])
        dataset = dataset.map(lambda x,y: (data_augment(x, training=True), y))

    # Chuẩn hóa theo ResNet50
    dataset = dataset.map(lambda x, y: (preprocess_input(x), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset(TRAIN_DIR, augment=True, shuffle=True)
val_ds = create_dataset(VAL_DIR, augment=False, shuffle=False)


# ==============================
# 🔹 Xây dựng model ResNet50
# ==============================
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Freeze 80% lớp đầu + BatchNormalization layers
    for layer in base_model.layers[:-50]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = False

    for layer in base_model.layers[-50:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Compile với label smoothing
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================
# 🔹 huấn luyện mô hình
# ==============================
if __name__ == "__main__":
    model = build_model()
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

   
    num_train = sum([len(files) for r,d,files in os.walk(TRAIN_DIR) if files])
    num_val = sum([len(files) for r,d,files in os.walk(VAL_DIR) if files])

    steps_per_epoch = BATCH_SIZE
    if num_train % BATCH_SIZE != 0: steps_per_epoch += 1

    validation_steps = BATCH_SIZE
    if num_val % BATCH_SIZE != 0: validation_steps += 1

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb, csv_logger_cb, tensorboard_cb]
    )