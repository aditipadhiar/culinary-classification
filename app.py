from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid

app = Flask(__name__)

# Load the saved model
model_path = 'culinary_resnet50.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()
resnet_model = load_model(model_path)

# Define label meanings
label_meanings = {
    0: 'French fries',
    1: 'Fried rice',
    2: 'Nachos',
    3: 'Salad',
    4: 'Spaghetti'
}

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    #img_tensor /= 255.
    return img_tensor

@app.route('/', methods=['GET'])
def index():
    print("get method call")
    return "Hello"

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file is missing or empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the file temporarily with a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    temp_file_path = f"temp_{uuid.uuid4().hex}{file_extension}"
    file.save(temp_file_path)
    
    # Preprocess the input image
    input_data = preprocess_image(temp_file_path)
    
    # Make predictions
    softmax_score = resnet_model.predict(input_data)
    
    # Get the predicted class
    predicted_class_index = np.argmax(softmax_score)
    predicted_class = label_meanings[predicted_class_index]
    
    # Prepare the response
    response = {
        'predicted_class': predicted_class
    }
    
    # Remove the temporary file
    os.remove(temp_file_path)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
