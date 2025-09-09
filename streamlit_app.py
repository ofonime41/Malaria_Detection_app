# ...existing code...
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from tensorflow.keras.applications.efficientnet import preprocess_input
# ...existing code...

# Load your trained model
@st.cache_resource  # ✅ caches model so it doesn't reload every time
def load_model():
    # try loading the model file
    m = tf.keras.models.load_model("efficientnetb0_model2.keras", compile=False)

    # determine model's expected input shape
    in_shape = m.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    _, h, w, c = in_shape

    # if the saved model expects a single-channel input but app will provide RGB,
    # wrap the loaded model with a small preprocessing layer that converts RGB->grayscale
    if c == 1:
        # build wrapper that accepts RGB and converts to grayscale for the inner model
        new_input = tf.keras.Input(shape=(h or 224, w or 224, 3), name="rgb_input")
        # convert RGB to grayscale (keeps a trailing channel dim)
        x = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x), name="rgb_to_gray")(new_input)
        outputs = m(x)
        wrapper = tf.keras.Model(new_input, outputs, name="wrapped_model_rgb_to_gray")
        return wrapper

    # if the model expects 3 channels, return it as-is
    return m

model = load_model()
# ...existing code...

# determine target size from model input (fallback to 224)
_in_shape = model.input_shape
if isinstance(_in_shape, list):
    _in_shape = _in_shape[0]
TARGET_HEIGHT = _in_shape[1] or 224
TARGET_WIDTH = _in_shape[2] or 224
# ...existing code...

# Define class names (check train_generator.class_indices during training)
class_names = ["Malaria Found", "Normal - No Malaria"]

# Preprocess function
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")  # ensures 3 channels
    img = image.resize((TARGET_WIDTH, TARGET_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # If inner model expects grayscale, wrapper already handles conversion.
    # Ensure array has 3 channels before preprocess_input when appropriate.
    if img_array.shape[-1] == 1 and model.input_shape[-1] == 3:
        # expand single channel to 3 by repeating
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ...existing code...
# Streamlit UI
st.title("🦟 Malaria Detection App")
st.write("Upload a blood smear image to check for malaria.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Always load as RGB to avoid grayscale issues
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image with smaller size
    st.image(image, caption="Uploaded Image", use_container_width=False, width=300)

    # Progress bar simulation
    st.write("🔍 Running model prediction...")
    progress_bar = st.progress(0)
    for percent in range(0, 101, 20):
        time.sleep(0.2)  # simulate work
        progress_bar.progress(percent, text=f"Processing... {percent}%")

    # Preprocess and predict
    img_array = preprocess_image(image)
    st.write("Image shape before prediction:", img_array.shape)

    prediction = model.predict(img_array)

    # Pick highest confidence class
    pred_class = np.argmax(prediction[0])
    confidence = prediction[0][pred_class]
    result = class_names[pred_class]

    # Display results
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2%}")
# ...existing code...