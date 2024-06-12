# Animal Image Recognition Project

## Overview
This project classifies images of animals into 10 categories: dog, horse, elephant, butterfly, chicken, cat, cow, sheep, squirrel, and spider.

## Directory Structure
- `data/`: Contains datasets.
- `models/`: Stores trained models.
- `main.py`: Script to train the model.
- `ui.py`: Script for the tkinter user interface.
- `README.md`: Overview and instructions.
- `requirements.txt`: Project dependencies.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mayokun-Sofowora/Animal-Image-Recognition.git
   cd project


pip install venv
.venv\Scripts\activate # This command should be run Initially to create the virtual environment command line.
pip install -r requirements.txt


After making changes 

pip freeze > requirements.txt
deactivate


## The Goal of The Project

1. **Define the scope of the project**:
The objective is to recognize different animal species from images.
Input: Images of animals.
Output: Predicted animal species.

2. **Data collection and preparation**:
Collect a dataset of animal images (e.g. from Kaggle).
Preprocessed the images (resize, normalize, etc.).
Split the dataset into training, validation, and test sets.

3. **Build the neural network model**:
Design a Convolution Neural Network (CNN) for image recognition.
Implement the model using a deep learning framework like TensorFlow or PyTorch.

4. **Implement evolutionary algorithm**:
Use evolutionary algorithms to optimize the neural network architecture and hyperparameter.
Implement the genetic algorithm (GA) to evolve the best model configurations.

5. **Train and evaluate the model**:
Train the CNN model on the training dataset.
Use the validation dataset to fine-tune the model.
Evaluate the model performance on the test dataset.

6. **Deploy the model**:
Deploy the trained model for interface.
Create a simple user interface to upload images and display predictions.

7. **Documentation and reporting**:
Document the code and methodology.
Prepare a report and presentation detailing the project.

