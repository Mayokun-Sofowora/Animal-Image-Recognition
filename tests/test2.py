import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pywt

# DEFINE CONSTANTS
img_size = 100
categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
              "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
classes = len(animals)
inputShape = (img_size, img_size, 1)
modelName = "test_model"
modelsFolderPath = "../models/"
modelPath = modelsFolderPath + modelName + '/'


def preprocess_data():
    data = []
    for category, translate in categories.items():
        path = '../data/animals10/raw-img/' + category
        target = animals.index(translate)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_img_array = cv2.resize(img_array, (img_size, img_size))
                data.append([new_img_array, target])
            except Exception as e:
                pass
    return data


def build_model():
    model = models.Sequential()
    model.add(layers.AveragePooling2D((5, 4), input_shape=inputShape))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()])
    return model


if __name__ == "__main__":
    # PREPROCESS THE DATA
    data = preprocess_data()
    random.shuffle(data)
    x = []
    y = []
    for features, labels in data:
        x.append(features)
        y.append(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=classes)
    x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=classes)

    # Ensure the model directory exists
    os.makedirs(modelPath, exist_ok=True)

    # BUILD AND COMPILE MODEL
    model = build_model()
    model.summary()

    # DEFINE CALLBACKS
    ReduceLROnPlateau_callback = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=60,
        verbose=1,
        factor=0.3,
        min_lr=0.0000001)

    ModelCheckpoint_callback = callbacks.ModelCheckpoint(
        modelPath + 'mdl_wts.keras',  # Updated file extension to .keras
        save_best_only=True,
        monitor='val_accuracy',
        mode='max')

    EarlyStopping_callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        start_from_epoch=20,
        verbose=1,
        mode='min')

    # TRAIN THE MODEL
    history = model.fit(x_train, y_train, epochs=10, batch_size=256, validation_split=0.2,
                        callbacks=[ReduceLROnPlateau_callback, ModelCheckpoint_callback, EarlyStopping_callback])

    # EVALUATE THE MODEL
    loss, accuracy, precision, recall, auc = model.evaluate(x_test, y_test)
    print(f"TEST LOSS: {loss}")
    print(f"TEST ACCURACY: {accuracy}")
    print(f"TEST PRECISION: {precision}")
    print(f"TEST RECALL: {recall}")
    print(f"TEST AUC: {auc}")

    # CONFUSION MATRICES
    y_train_pred = model.predict(x_train).argmax(axis=1)
    y_test_pred = model.predict(x_test).argmax(axis=1)
    y_train_true = y_train.argmax(axis=1)
    y_test_true = y_test.argmax(axis=1)

    cm_train = confusion_matrix(y_train_true, y_train_pred)
    cm_test = confusion_matrix(y_test_true, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Train Confusion Matrix')
    sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[1], cmap='Blues')
    axes[1].set_title('Test Confusion Matrix')

    plt.tight_layout()
    plt.show()

    # PLOT TRAINING AND VALIDATION ACCURACY AND LOSS
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Train Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # MORLE WAVELENGTH SCALOGRAM
    # Example: generating scalogram for a row in the first image of x_train
    image_row = x_train[0, :, 0, 0]  # First row of the first training image
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(image_row, scales, 'morl')

    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(coefficients), extent=[0, len(image_row), scales[-1], scales[0]], cmap='jet', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet Scalogram')
    plt.show()

    print("\nMORL Wavelength Scalogram: (Generated for a sample row of the first training image)")
