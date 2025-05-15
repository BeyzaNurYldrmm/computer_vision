import os
import matplotlib.pyplot as plt
import random
import warnings
import shutil
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

base_data_dir = r"!!veri seti yolu!!"
image_folders = {
    'glioma': os.path.join(base_data_dir, 'brain_glioma'),
    'meningioma': os.path.join(base_data_dir, 'brain_menin'),
    'pituitary': os.path.join(base_data_dir, 'brain_tumor')
}
base_dir = r"!!bu dosyaların kaydedileceği yol"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

for split_dir in [train_dir, test_dir]:
    os.makedirs(split_dir, exist_ok=True)
    for label in image_folders:
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)

for label, folder_path in image_folders.items():
    images = os.listdir(folder_path)
    train_imgs, test_imgs = train_test_split(images, test_size=0.3, random_state=42)
    for img in train_imgs:
        shutil.copy(os.path.join(folder_path, img), os.path.join(train_dir, label, img))
    for img in test_imgs:
        shutil.copy(os.path.join(folder_path, img), os.path.join(test_dir, label, img))
labels_mapping = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'pituitary': 'Other Tumor'
}

all_images = []
for folder in image_folders:
    label = labels_mapping[folder]
    folder_path = image_folders[folder]
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append((os.path.join(folder_path, filename), label))

num_images_to_show = 10
random_images = random.sample(all_images, min(num_images_to_show, len(all_images)))

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()
for i, (image_path, label) in enumerate(random_images):
    image = plt.imread(image_path)
    axes[i].imshow(image)
    axes[i].set_title(label, color='red')
    axes[i].axis('off')
plt.tight_layout()
plt.show()

label_counts = {v: 0 for v in labels_mapping.values()}
for folder_key in image_folders:
    folder_label = labels_mapping[folder_key]
    for filename in os.listdir(image_folders[folder_key]):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_counts[folder_label] += 1

plt.figure(figsize=(10, 6))
plt.bar(label_counts.keys(), label_counts.values(), color=['skyblue', 'salmon', 'lightgreen'])
plt.xlabel('Tümör Tipi')
plt.ylabel('Görsel Sayısı')
plt.title('Her Tümör Tipine Ait Görsel Dağılımı')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

label_sizes = {v: 0 for v in labels_mapping.values()}
label_counts = {v: 0 for v in labels_mapping.values()}
for key, folder in image_folders.items():
    label = labels_mapping[key]
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            try:
                label_sizes[label] += os.path.getsize(path)
                label_counts[label] += 1
            except FileNotFoundError:
                print(f"Not found: {path}")

average_sizes = {
    label: label_sizes[label] / label_counts[label] if label_counts[label] > 0 else 0
    for label in label_sizes
}

plt.figure(figsize=(10, 6))
plt.bar(average_sizes.keys(), average_sizes.values(), color=['orange', 'cyan', 'purple'])
plt.xlabel('Tümör Tipi')
plt.ylabel('Ortalama Dosya Boyutu (Bytes)')
plt.title('Ortalama Görsel Boyutu')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tqdm.keras import TqdmCallback
import numpy as np

image_size = (224, 224)
batch_size = 32
epochs = 30
num_classes = 3

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))

inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[TqdmCallback(verbose=1), reduce_lr, early_stopping],
    class_weight=class_weight_dict
)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu', marker='o')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu', marker='x')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı', marker='o')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı', marker='x')
plt.title('Model Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
