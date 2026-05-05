import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shutil # You don't use this, but it's in your original imports

# --- Flask App Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- Model and Classes Configuration ---
# Ensure these files and folders are in the same directory as your app.py
MODEL_PATH = 'pollen_model.keras'
ORGANIZED_DATASET_DIR = 'organized_dataset'
TARGET_SIZE = (128, 128)

# Global variables to hold the loaded model and class names
model = None
class_names = []

# --- Function to load resources once when the app starts ---
def load_resources():
    """Loads the Keras model and class names into global variables."""
    global model, class_names
    try:
        print("Loading model...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        
        if not os.path.exists(ORGANIZED_DATASET_DIR):
            print(f"Error: The directory '{ORGANIZED_DATASET_DIR}' does not exist.")
            print("Please ensure this folder contains the subfolders with your class names.")
            # Set a flag or handle the error appropriately
            class_names = [] # Set to empty list to avoid errors later
        else:
            class_names = sorted(os.listdir(ORGANIZED_DATASET_DIR))
            print(f"Detected {len(class_names)} classes: {class_names}")

    except Exception as e:
        print(f"Error loading resources: {e}")
        model = None # Set to None to indicate loading failed
        class_names = []
        # You might want to shut down the app or show a maintenance page in a real-world scenario.

# --- Core Prediction Logic ---
def predict_single_image(img_path):
    """
    Predicts the class of a single image.
    
    Args:
        img_path (str): The file path to the image.
        
    Returns:
        tuple: A tuple containing (predicted_class, confidence_percentage) or (None, None) if an error occurs.
    """
    if model is None or not class_names:
        print("Model or class names not loaded. Cannot predict.")
        return "Model not loaded.", None

    try:
        # Preprocess the image
        img = image.load_img(img_path, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make the prediction
        prediction = model.predict(img_array)
        
        # Get the predicted class and confidence
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Renders the main upload page and handles file uploads."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Pass the URL path for the image to the next page
            image_url = url_for('static', filename=f'uploads/{filename}')
            return redirect(url_for('show_prediction_page', image_path=image_url))
    
    # Renders the styled index.html page
    return render_template('index.html')

@app.route('/predict_page')
def show_prediction_page():
    """Renders the page with the uploaded image and a predict button."""
    image_path = request.args.get('image_path')
    if not image_path:
        return redirect(url_for('upload_file'))
    return render_template('prediction.html', image_path=image_path)

@app.route('/predict', methods=['POST'])
def predict_image_class():
    """Handles the prediction request and displays the result."""
    # Get the image path from the form's hidden input
    image_url = request.form.get('image_path')
    
    # Convert the URL path back to a file path on disk
    filename = os.path.basename(image_url)
    file_path_on_disk = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not file_path_on_disk or not os.path.exists(file_path_on_disk):
        return "Error: Image not found for prediction.", 404
        
    # Call the core prediction logic function
    predicted_class, confidence = predict_single_image(file_path_on_disk)
    
    if predicted_class is not None:
        result_message = f"✅ Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)"
    else:
        result_message = "❌ Prediction failed. Please check the logs."
    
    # Render the result page with the prediction outcome
    return render_template('logout.html', result=result_message, image_path=image_url)

if __name__ == '__main__':
    # Create the upload directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Load the model and classes before starting the server
    load_resources()
    
    # Run the Flask application
    app.run(debug=True)