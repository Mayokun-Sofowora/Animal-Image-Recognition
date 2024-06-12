import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# LOAD CONSTANTS
IMAGE_SIZE = (100, 100)
MODEL_PATH = './models/animal_recognition.keras'
CATEGORIES = ['Dog', 'Horse', 'Elephant', 'Butterfly', 'Chicken', 'Cat', 'Cow', 'Sheep', 'Squirrel', 'Spider']


def load_model(model_path):
    """
    Load the trained model
    :param model_path: The path to the saved model
    :return: Returns the loaded model or None if failed
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the model: {e}")
        return None


def prepare_image(filepath):
    """
    Prepare the image for prediction
    :param filepath: The path to the image file
    :return: Returns the array of the new image or None if failed
    """
    try:
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, IMAGE_SIZE)
        new_img_array = new_img_array.reshape(-1, *IMAGE_SIZE, 1)
        new_img_array = tf.keras.utils.normalize(new_img_array, axis=1)
        return new_img_array
    except Exception as e:
        messagebox.showerror("Error", f"Failed to prepare image: {e}")
        return None


def load_and_predict():
    """
    Load an image and make a prediction
    :return: Returns None
    """
    filepath = filedialog.askopenfilename()
    if filepath:
        try:
            img = Image.open(filepath)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)

            panel = tk.Label(root, image=img)
            panel.image = img
            panel.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

            img_array = prepare_image(filepath)
            if img_array is not None:
                prediction = model.predict(img_array)
                predicted_class = CATEGORIES[np.argmax(prediction)]
                messagebox.showinfo("Prediction", f"The predicted class is: {predicted_class}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load and predict: {e}")


# INITIALIZE TKINTER WINDOW
root = tk.Tk()
root.title("Animal Image Recognition")

# LOAD THE MODEL
model = load_model(MODEL_PATH)

if __name__ == '__main__':
    # LOAD THE MODEL
    model = load_model(MODEL_PATH)

    if model:
        # SET BACKGROUND COLOUR
        root.configure(background="lightblue")

        # TITLE LABEL
        title_label = tk.Label(root, text="Animal Image Recognition", font=("Lato", 18, "bold"), bg="lightblue")
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # INSTRUCTIONS LABEL
        instructions_label = tk.Label(root, text="Click 'Load Image' to select an image.", font=("Lato", 12), bg="lightblue")
        instructions_label.grid(row=1, column=0, columnspan=2, pady=5)

        # CREATE AND PLACE THE BUTTONS
        load_button = tk.Button(root, text="Load Image", command=load_and_predict, width=15, font=("Lato", 12), bg="orange", fg="white")
        load_button.grid(row=2, column=0, padx=10, pady=10)

        quit_button = tk.Button(root, text="Quit", command=root.quit, width=15, font=("Lato", 12), bg="red", fg="white")
        quit_button.grid(row=2, column=1, padx=10, pady=10)

        # RUN THE TKINTER MAIN LOOP
        root.mainloop()
