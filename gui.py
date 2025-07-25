import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Set the environment variable to disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained model (Ensure your model is in the 'model/' directory or update the path)
model = load_model('model/crop_model.h5')

# Class names for the model (modify based on your model's classes)
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
        img = Image.open(image_path).resize((28, 28))  # Resize based on model requirements
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = img_array.reshape(1, 28, 28, 3)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]  # Get the highest probability class
        confidence = np.max(predictions) * 100  # Confidence percentage

        return predicted_class, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0.0


def upload_and_predict():
    """
    Opens a file dialog to upload an image and predicts the crop health.
    Displays the result on the GUI.
    """
    # Open file dialog to upload an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        try:
            # Open and resize the image for display in the GUI
            img = Image.open(file_path).resize((250, 250))  # Resize for GUI display
            img_display = ImageTk.PhotoImage(img)
            
            # Display the image
            img_label.config(image=img_display)
            img_label.image = img_display

            # Predict crop health and display the result
            predicted_class, confidence = predict_crop_health(file_path)
            if confidence == 0.0:  # Handle error in prediction
                result_label.config(text=f"Prediction Error: {predicted_class}")
            else:
                result_label.config(text=f"Result: {predicted_class} (Confidence: {confidence:.2f}%)")
        
        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")


# Create the main GUI window
root = tk.Tk()
root.title("AI Crop Health Monitoring")

# Add the "Upload Image" button
upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_btn.pack()

# Label to display the uploaded image
img_label = tk.Label(root)
img_label.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="Result: ", font=("Arial", 14))
result_label.pack()

# Run the GUI main loop
root.mainloop()
