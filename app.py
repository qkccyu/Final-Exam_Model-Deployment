import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_classifier.hdf5')
        return model
    except OSError:
        st.error("Model file 'plant_classifier.hdf5' not found. Please ensure the file is in the correct path.")

model = load_model()

st.write("""
# Weather
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Updated to use LANCZOS
    img = np.asarray(image)
    if img.ndim == 2:  # Grayscale image
        img = np.stack((img,) * 3, axis=-1)  # Convert to RGB
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_container_width=True)  # Updated to use_container_width
        prediction = import_and_predict(image, model)
        
        # Updated class names for weather conditions
        class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        st.success(string)
    except Exception as e:
        st.error(f"Error processing the image: {e}")
