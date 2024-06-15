import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pywt

# DEFINE CONSTANTS
img_size = 100
categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
              "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]


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


# BUILD MODEL
def build_model(with_batch_norm=False, kernel_size=3, dropout_rate=0.2):
    model = Sequential([
        Conv2D(32, kernel_size=kernel_size, input_shape=x_train.shape[1:]),
        Activation('relu')
    ])
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=kernel_size))
    model.add(Activation('relu'))
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=kernel_size, strides=2, padding='same'))
    model.add(Activation('relu'))
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(64, kernel_size=kernel_size, strides=2, padding='same'))
    model.add(Activation('relu'))
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(256, kernel_size=kernel_size))
    model.add(Activation('relu'))
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(Activation('relu'))
    if with_batch_norm:
        model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# TRAIN THE MODEL
def train_and_evaluate(with_batch_norm=False, epochs=5, kernel_size=3, dropout_rate=0.2):
    model = build_model(with_batch_norm, kernel_size, dropout_rate)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=256, validation_split=0.2)
    loss, accuracy = model.evaluate(x_test, y_test)
    return history, loss, accuracy, model


if __name__ == "__main__":
    # TEST WITH PARAMETERS
    epochs = 10
    kernel_size = 3
    dropout_rate = 0.4

    print('\n-----------------------------------------------------------------------------------------')
    print(f"TESTING WITH EPOCHS = {epochs}, KERNEL_SIZE = {kernel_size}, DROPOUT_RATE = {dropout_rate}")
    print('-----------------------------------------------------------------------------------------\n')

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
    y_train = np.array(y_train)
    x_test = np.array(x_test).reshape(-1, img_size, img_size, 1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    y_test = np.array(y_test)

    # TRAIN THE MODEL
    history_no_bn, loss_no_bn, accuracy_no_bn, model_no_bn = train_and_evaluate(with_batch_norm=False, epochs=epochs,
                                                                                kernel_size=kernel_size,
                                                                                dropout_rate=dropout_rate)
    history_bn, loss_bn, accuracy_bn, model_bn = train_and_evaluate(with_batch_norm=True, epochs=epochs,
                                                                    kernel_size=kernel_size,
                                                                    dropout_rate=dropout_rate)

    # MODEL SUMMARY
    print("\nModel Summary without Batch Normalization:")
    model_no_bn.summary()
    print("\nModel Summary with Batch Normalization:")
    model_bn.summary()

    # CONFUSION MATRICES
    y_train_pred_no_bn = model_no_bn.predict(x_train).argmax(axis=1)
    y_test_pred_no_bn = model_no_bn.predict(x_test).argmax(axis=1)
    y_train_pred_bn = model_bn.predict(x_train).argmax(axis=1)
    y_test_pred_bn = model_bn.predict(x_test).argmax(axis=1)

    cm_train_no_bn = confusion_matrix(y_train, y_train_pred_no_bn)
    cm_test_no_bn = confusion_matrix(y_test, y_test_pred_no_bn)
    cm_train_bn = confusion_matrix(y_train, y_train_pred_bn)
    cm_test_bn = confusion_matrix(y_test, y_test_pred_bn)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.heatmap(cm_train_no_bn, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Train Confusion Matrix without BN')
    sns.heatmap(cm_test_no_bn, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
    axes[0, 1].set_title('Test Confusion Matrix without BN')
    sns.heatmap(cm_train_bn, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
    axes[1, 0].set_title('Train Confusion Matrix with BN')
    sns.heatmap(cm_test_bn, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title('Test Confusion Matrix with BN')

    plt.tight_layout()
    plt.show()

    # PLOT TRAINING AND VALIDATION ACCURACY AND LOSS
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_no_bn.history['accuracy'], label='Train Accuracy without BN')
    plt.plot(epochs_range, history_no_bn.history['val_accuracy'], label='Validation Accuracy without BN')
    plt.plot(epochs_range, history_bn.history['accuracy'], label='Train Accuracy with BN')
    plt.plot(epochs_range, history_bn.history['val_accuracy'], label='Validation Accuracy with BN')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_no_bn.history['loss'], label='Train Loss without BN')
    plt.plot(epochs_range, history_no_bn.history['val_loss'], label='Validation Loss without BN')
    plt.plot(epochs_range, history_bn.history['loss'], label='Train Loss with BN')
    plt.plot(epochs_range, history_bn.history['val_loss'], label='Validation Loss with BN')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    print(f"TEST LOSS WITHOUT BATCH NORMALIZATION: {loss_no_bn}")
    print(f"TEST ACCURACY WITHOUT BATCH NORMALIZATION: {accuracy_no_bn}")
    print(f"TEST LOSS WITH BATCH NORMALIZATION: {loss_bn}")
    print(f"TEST ACCURACY WITH BATCH NORMALIZATION: {accuracy_bn}")

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
