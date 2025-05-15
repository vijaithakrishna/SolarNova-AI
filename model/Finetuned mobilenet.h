import time
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import random
from tqdm import tqdm
import os
import shutil  # Add this import statement
import numpy as np

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

data_dir = r"C:\Users\shaha\Downloads\dataset"
clean_images_dir = os.path.join(data_dir, 'clean')  # Use os.path.join for directory paths
dirty_images_dir = os.path.join(data_dir, 'dirt')   # Use os.path.join for directory paths

random.seed(101)
SIZE = (224, 224)
BATCH_SIZE = 64
RANDOM_STATE = 101
GRAYSCALE = False
if GRAYSCALE == True:
    INPUT_SHAPE = SIZE + (1,)
else:
    INPUT_SHAPE = SIZE + (3,)

def load_images(directory, grayscale=GRAYSCALE, size=SIZE, use_augmented=True):
    destination_dir = directory + "_Discarded"
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    images = []  # Creating a list to store the images in the form of array
    for filename in tqdm(os.listdir(directory)):
        img = cv2.imread(os.path.join(directory, filename))
        try:
            height, width, _ = img.shape
            # move images with small dimensions to the discarded folder
            if (height < 800 and width < 400) or (height < 400 and width < 800):
                shutil.move(os.path.join(directory, filename), os.path.join(destination_dir, filename))
                continue

            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, size)
            x = np.array(img)
            images.append(x)
        except Exception as e:
            continue

    if use_augmented:
        directory = os.path.join(directory)
        for filename in tqdm(os.listdir(directory)):
            img = cv2.imread(os.path.join(directory, filename))
            try:
                height, width, _ = img.shape
                # move images with small dimensions to the discarded folder
                if (height < 800 and width < 400) or (height < 400 and width < 800):
                    shutil.move(os.path.join(directory, filename), os.path.join(destination_dir, filename))
                    continue

                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, size)
                x = np.array(img)
                images.append(x)
            except Exception as e:
                continue

    return np.array(images)  # Convert the list to numpy array


clean_images_array = load_images(clean_images_dir, grayscale=GRAYSCALE)
dusty_images_array = load_images(dirty_images_dir, grayscale=GRAYSCALE)

# Assuming you have labels for your images as well (0 for clean, 1 for dusty)
class1_labels = np.zeros(len(clean_images_array))
class2_labels = np.ones(len(dusty_images_array))

# Concatenate images and labels
x_train = np.concatenate((clean_images_array, dusty_images_array), axis=0)
y_train = np.concatenate((class1_labels, class2_labels), axis=0)

# Shuffle the data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# Split data into training and validation sets
split_ratio = 0.8  # 80% training, 20% validation
split_index = int(split_ratio * len(x_train))

x_val = x_train[split_index:]
y_val = y_train[split_index:]
x_train = x_train[:split_index]
y_train = y_train[:split_index]

# Step 1: Load the pre-trained model
pretrained_model_path = r"C:\Users\shaha\OneDrive\Desktop\project\dataset\Mobilenet.h5"
model = load_model(pretrained_model_path)

# Step 4: Fine-tune the model
# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))

# Step 5: Save the fine-tuned model
fine_tuned_model_path = r"C:\Users\shaha\OneDrive\Desktop\project\dataset\FineTuned_Mobilenet.h5"
model.save(fine_tuned_model_path)
