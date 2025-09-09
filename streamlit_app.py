import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# Load your trained model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb0_model2.keras", compile=False)

model = load_model()

# -------------------------------
# Define target size (match model)
# -------------------------------
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# Define class names
class_names = ["Malaria Found", "Normal - No Malaria"]

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    image = image.convert("RGB")
    img = image.resize((TARGET_WIDTH, TARGET_HEIGHT))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🦟 Malaria Detection App")
st.write("Upload a blood smear image to check for malaria.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

        # Progress bar
        st.write("🔍 Running model prediction...")
        progress_bar = st.progress(0)
        for percent in range(0, 101, 20):
            time.sleep(0.1)
            progress_bar.progress(percent, text=f"Processing... {percent}%")

        # Preprocess and predict
        img_array = preprocess_image(image)
        st.write("Debug - Image shape:", img_array.shape)

        prediction = model.predict(img_array, verbose=0)
        pred_class = np.argmax(prediction[0])
        confidence = prediction[0][pred_class]
        result = class_names[pred_class]

        # Display results
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
