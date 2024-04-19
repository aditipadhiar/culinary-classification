# Classify Culinary Items using Deep Learning

This project aims to classify culinary items using deep learning techniques, specifically leveraging the ResNet50 model. The model is trained on a dataset containing images of five different culinary items: French fries, Fried rice, Nachos, Salad, and Spaghetti. Each class has 300 images, resulting in a balanced dataset for training.

## Dataset
The dataset used for training is obtained from Kaggle, named Food 101. From this dataset, we selected five classes relevant to culinary items and extracted 300 images per class for training the model.

## Model Architecture
The ResNet50 architecture is utilized for training the classification model. ResNet50 is a deep convolutional neural network known for its effectiveness in image classification tasks. The pre-trained weights of ResNet50 are fine-tuned on the culinary dataset to improve classification accuracy.

## Training
The model is trained using the TensorFlow framework with the Keras API. The training process involves feeding the images into the model and adjusting the model's weights based on the prediction errors. The training is performed on a GPU-enabled machine to expedite the process.

## Deployment
The trained model is saved in the HDF5 format (.h5) and deployed using Flask, a web framework for Python. An API is created to serve predictions based on input images. The API endpoint accepts image files and returns the predicted class label for each image.

## Usage
To use the trained model for classification, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies specified in the requirements.txt file.
3. Run the Flask application by executing python app.py.
4. Send POST requests to the /predict endpoint of the API with image files attached.
5. Receive predictions in JSON format containing the predicted class labels for the uploaded images.

## Dependencies
* Python 3.x
* TensorFlow
* Keras
* Flask
* NumPy
* Matplotlib (optional, for visualization)
