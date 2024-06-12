import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    return history, loss, accuracy


if __name__ == "__main__":
    # TEST WITH DIFFERENT VALUES
    epochs_values = [5, 10, 20]
    kernel_sizes = [3, 4, 5]
    dropout_rates = [0.2, 0.4, 0.8]

    for epochs in epochs_values:
        for kernel_size in kernel_sizes:
            for dropout_rate in dropout_rates:
                print('\n-----------------------------------------------------------------------------------------')
                print(f"TESTING WITH EPOCHS = {epochs}, KERNEL_SIZE = {kernel_size}, DROPOUT_RATE = {dropout_rate}")

                # PREPROCESS THE DATA WITH THE CURRENT PARAMETERS
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

                # TRAIN THE MODEL WITH THE CURRENT PARAMETERS
                history_no_bn, loss_no_bn, accuracy_no_bn = train_and_evaluate(with_batch_norm=False, epochs=epochs,
                                                                               kernel_size=kernel_size,
                                                                               dropout_rate=dropout_rate)
                history_bn, loss_bn, accuracy_bn = train_and_evaluate(with_batch_norm=True, epochs=epochs,
                                                                      kernel_size=kernel_size,
                                                                      dropout_rate=dropout_rate)

                # PLOT THE RESULTS
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
