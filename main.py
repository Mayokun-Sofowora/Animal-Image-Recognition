import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import os
import random
import cv2
from sklearn.model_selection import train_test_split
from data.download_data import download_data
from data.preprocess_data import preprocess_images
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# DOWNLOAD THE DATASET FROM KAGGLE
if not os.path.exists('./data/animals10/raw-img'):
    download_data()

# PATH TO THE DATASET
# img_dir = './data/animals10/raw-img'
# preprocessed_images_dir = './data/preprocessed-img'
#
# # PREPROCESS THE IMAGES
# preprocess_images(img_dir, preprocessed_images_dir)

# MAPPING DIRECTORIES FROM DATASET TO ENGLISH LABELS
categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
              "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
data = []
animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
img_size = 100


# DOWNLOADING IMAGES FROM DATASET AND ASSIGNING THEM LABELS
def create_data():
    print("DOWNLOADING DATA FROM DATASET: START")

    # FOR EACH ANIMAL CATEGORY (categories.items()) FETCHES THE CATEGORY NAME AND ITS TRANSLATION
    for category, translate in categories.items():
        path = './data/animals10/raw-img/' + category
        target = animals.index(translate)
        print("DOWNLOADING ANIMALS FROM:" + translate)

        # INTERNAL LOOP, WHICH FOR EACH CATEGORY FETCHES THE NEXT IMAGES FROM THE DOWNLOADED PATH
        for img in os.listdir(path):
            try:
                # CONVERTING IMAGE FROM 'NORMAL' TO GRAYSCALE ARRAY
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                # RESIZING IMAGE TO 100X100 (ALTHOUGH THIS IS STILL TOO MUCH, SMALLER ONES SHOULD BE FASTER)
                new_img_array = cv2.resize(img_array, (img_size, img_size))

                # ADDING IMAGE AND ITS LABEL TO THE SET
                data.append([new_img_array, target])
            except Exception as e:
                pass
    print("DOWNLOADING DATA FROM DATASET: END")


create_data()

print("PREPARING TEST AND CONTROL SETS: START")

# SHUFFLING ELEMENTS IN THE SET
random.shuffle(data)
x = []
y = []
for features, labels in data:
    x.append(features)
    y.append(labels)

# DIVIDING THE SET INTO TEST AND TRAIN DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# UNDEFINED NORMALIZATION
x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
y_train = np.array(y_train)
print("PREPARING TEST AND CONTROL SETS: END")

# NEURAL NETWORK OPERATION
print("NOW OPERATING NEURAL NETWORK")


def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(img_size, img_size, 1)),
        Conv2D(32, kernel_size=3, activation='relu'),
        Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
        Dropout(0.2),
        Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
        Dropout(0.2),
        Conv2D(256, kernel_size=3, activation='relu'),
        Flatten(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


model = create_model()

history = model.fit(x_train, y_train, epochs=5, batch_size=256)
x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)
model.evaluate(x_test, y_test)

# Plotting training and validation accuracy and loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# SAVE MODELS
if not os.path.exists('./models'):
    os.makedirs('./models')
model.save('./models/animal_recognition.keras')
