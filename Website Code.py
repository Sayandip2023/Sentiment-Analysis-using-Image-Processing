import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import time

# Load the trained model
model = tf.keras.models.load_model('Sentiment.h5')

# Define class labels
classes = ['Disappointed', 'interested', 'neutral']

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Streamlit application
st.title("Sentiment Analysis using Computer Vision")
st.write("Upload an image to get sentiment classification")

uploaded_file = st.file_uploader("Upload the image here", type="jpg")

if uploaded_file is not None:
    # Load and preprocess image
    image = load_img(uploaded_file)
    preprocessed_image = preprocess_image(image)

    # Predict
    start_time = time.time()
    predictions = model.predict(preprocessed_image)
    inference_time = time.time() - start_time

    # Display results
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Inference Time: {inference_time:.4f} seconds")

    predicted_class = np.argmax(predictions, axis=1)[0]
    st.write(f"Detected emotion: {classes[predicted_class]}")
