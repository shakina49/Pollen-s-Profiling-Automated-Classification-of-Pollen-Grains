import zipfile
import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image

# Step 1: Extract archive.zip (place this zip in the same directory)
archive_path = 'archive.zip'
dataset_path = 'dataset'
organized_path = 'organized_dataset'

if not os.path.exists(dataset_path):
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("✅ Successfully extracted archive.zip to dataset/")
else:
    print("ℹ️ Dataset already extracted.")

# Step 2: Organize images by class
try:
    os.makedirs(organized_path, exist_ok=True)
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            class_name = filename.split("_")[0].lower().strip().replace("(", "").replace(")", "")
            class_dir = os.path.join(organized_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(dataset_path, filename)
            dst = os.path.join(class_dir, filename)
            shutil.move(src, dst)
    print("✅ Images organized into folders by class.")
except Exception as e:
    print(f"Error organizing images: {e}")

# Step 3: Visualize class distribution
names = [name.replace(' ', '_').split('_')[0] for name in os.listdir(organized_path) if os.path.isdir(os.path.join(organized_path, name))]
classes = Counter(names)
print("Class counts:", classes)

counts = [len(os.listdir(os.path.join(organized_path, folder))) for folder in sorted(os.listdir(organized_path))]
labels = sorted(os.listdir(organized_path))
plt.figure(figsize=(16, 5))
plt.bar(labels, counts, color='teal')
plt.xticks(rotation=90)
plt.title('Class Distribution')
plt.ylabel('Image Count')
plt.show()

# Step 4: Image size analysis
sizes = []
for folder in os.listdir(organized_path):
    folder_path = os.path.join(organized_path, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    sizes.append(img.shape[:2])
if sizes:
    x, y = zip(*sizes)
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.5)
    plt.title("Image Size Scatterplot")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.plot([0, 800], [0, 800], 'r-')
    plt.grid(True)
    plt.show()
else:
    print("No images found in the target directory for size analysis.")

# Step 5: Data Preparation
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    organized_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    organized_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Step 6: Model Building
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 7: Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# Step 8: Save the Model
model.save("pollen_model.keras")
print("✅ Model saved as pollen_model.keras")

MODEL_PATH = 'pollen_model.keras'
TEST_IMAGES_DIR = 'test_images'
ORGANIZED_DATASET_DIR = 'organized_dataset'

def predict_on_images(model_path, images_dir, class_dir):
    try:
        print("Loading the model...")
        model = load_model(model_path)
        print("Model loaded successfully.")

        if not os.path.exists(class_dir):
            print(f"Error: The directory '{class_dir}' does not exist.")
            print("Please ensure this folder contains the subfolders with your class names.")
            return
            
        class_names = sorted(os.listdir(class_dir))
        print(f"Detected {len(class_names)} classes: {class_names}")

        if not os.path.exists(images_dir):
            print(f"Error: The test images directory '{images_dir}' does not exist.")
            print("Please create this directory and place your images inside it.")
            return

        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            print(f"No image files found in the '{images_dir}' directory.")
            return

        for filename in image_files:
            img_path = os.path.join(images_dir, filename)

            try:
                img = image.load_img(img_path, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.

                prediction = model.predict(img_array)
                
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_names[predicted_class_index]
                confidence = np.max(prediction) * 100

                print(f"✅ Predicted Class for '{filename}': {predicted_class} (Confidence: {confidence:.2f}%)")

            except Exception as e:
                print(f"Error processing image '{filename}': {e}")
                
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    predict_on_images(
        model_path=MODEL_PATH,
        images_dir=TEST_IMAGES_DIR,
        class_dir=ORGANIZED_DATASET_DIR
    )