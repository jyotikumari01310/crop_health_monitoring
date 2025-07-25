from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('model/crop_model.h5')  # Ensure the path is correct

# Class names for the model (modify this based on your pre-trained model)
class_names = ['Healthy', 'Disease', 'Pest', 'Nutrient Deficiency']

def predict_crop_health(image_path):
    """
    Predict the crop health from the given image.

    Args:
        image_path (str): Path to the image to be classified.

    Returns:
        predicted_class (str): Predicted class label.
        confidence (float): Confidence of the prediction as a percentage.
    """
    try:
        # Load and preprocess the image
        img = Image.open(image_path).resize((224, 224))  # Resize based on model requirements
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = img_array.reshape(1, 224, 224, 3)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]  # Get the highest probability class
        confidence = np.max(predictions) * 100  # Confidence percentage

        return predicted_class, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0
