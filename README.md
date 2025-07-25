# 🌾 AI-Powered Crop Health Monitoring

This is a Python-based mini project that detects the health status of crops using an image-based prediction model. The user uploads a crop image through a GUI, and the system predicts whether the crop is **Healthy**, **Pest-Affected**, or **Nutrient Deficient** using a deep learning model.

---

## 🚀 Features

- Upload leaf images through a simple GUI
- AI model predicts crop condition
- Fast and lightweight – suitable for small systems
- Accurate predictions using a pre-trained `.h5` deep learning model

---

## 🛠️ Tech Stack

- Python 3.x
- Keras + TensorFlow
- Tkinter (for GUI)
- OpenCV or Pillow (for image handling)
- `.h5` model file (saved CNN model)

---

## 📁 File Descriptions

| File | Description |
|------|-------------|
| `backend.py` | Loads model and processes the uploaded image for prediction |
| `gui.py` | Provides a GUI interface for users to upload images |
| `crop_model.h5` | Pre-trained AI model for crop classification |
| `requirements.txt` | Python package dependencies |


---

