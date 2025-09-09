import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load your trained model
@st.cache_resource  # ✅ caches model so it doesn't reload every time
def load_model():
    return tf.keras.models.load_model("efficientnetb0_model.keras")

model = load_model()

# Define class names (check train_generator.class_indices during training)
class_names = ["Malaria Found ", "Normal- No Malaria"]  

# Preprocess function
def preprocess_image(image: Image.Image):
    img = image.resize((224, 224))  # EfficientNet input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # ✅ same as training
    return img_array

# Streamlit UI
st.title("🦟 Malaria Detection App")
st.write("Upload a blood smear image to check for malaria.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with smaller size
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)


    #Add a progress bar
    st.write("🔍 Running model prediction")
    progress_bar = st.progress(0)

    # Simulate progress (for effect)
    for percent in range(0, 101, 20):
        time.sleep(0.2)  # simulate work
        progress_bar.progress(percent, text=f"Processing... {percent}%")
    # Preprocess and predict
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)

    # Softmax: pick highest confidence
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class]
    result = class_names[pred_class]

    # Display results
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2%}")
    