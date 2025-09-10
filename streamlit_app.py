import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda

# -------------------------------
# Load your trained model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        # Create a new input layer with correct dimensions (224x224)
        inputs = Input(shape=(224, 224, 3), name='new_input')
        
        # Load the base model
        base_model = tf.keras.models.load_model("efficientnetb0_model2.keras", compile=False)
        
        # Use the functional API to create new model
        x = inputs
        # Add a preprocessing Lambda layer to handle any necessary conversions
        x = Lambda(lambda x: tf.cast(x, tf.float32))(x)
        x = base_model(x)
        
        # Create new wrapped model
        wrapped_model = Model(inputs=inputs, outputs=x, name="wrapped_malaria_model")
        return wrapped_model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise e

# Initialize model
try:
    model = load_model()
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()  # Stop app execution if model fails to load

# -------------------------------
# Define target size (match model)
# -------------------------------
TARGET_HEIGHT = 224
TARGET_WIDTH = 224

# ...existing code...
# Define class names
class_names = ["Malaria Found", "Normal - No Malaria"]

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    img = image.resize((TARGET_WIDTH, TARGET_HEIGHT))
    
    # Convert to array and add batch dimension
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply EfficientNet preprocessing
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
        st.write("Debug - Input shape:", img_array.shape)

        prediction = model.predict(img_array, verbose=0)
        pred_class = np.argmax(prediction[0])
        confidence = prediction[0][pred_class]
        result = class_names[pred_class]

        # Display results
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")