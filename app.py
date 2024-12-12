import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('plant_classifier.hdf5')
        return model
    except FileNotFoundError:
        st.error("Model file 'plant_classifier.hdf5' not found. Please ensure the file is in the correct path.")
        return None

model = load_model()

st.write("""
# Plant Leaf Detection System
""")

file = st.file_uploader("Choose plant photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    if img.ndim == 2:  # Grayscale image
        img = np.stack((img,) * 3, axis=-1)  # Convert to RGB
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if model is not None:
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            image = Image.open(file)
            st.image(image, use_column_width=True)
            prediction = import_and_predict(image, model)
            
            class_names = [
                'Rasna', 'Arive-Dantu', 'Jackfruit', 'Neem', 'Basale',
                'Indian Mustard', 'Karanda', 'Lemon', 'Roxburgh fig', 'Peepal Tree',
                'Hibiscus', 'Jasmine', 'Mango', 'Mint', 'Drumstick',
                'Jamaica Cherry', 'Curry Leaf', 'Oleander', 'Parijata', 'Tulsi',
                'Betel', 'Mexican Mint', 'Indian Beech', 'Guava', 'Pomegranate',
                'Sandalwood', 'Jamun', 'Rose Apple', 'Crape Jasmine', 'Fenugreek'
            ]
            
            string = "OUTPUT : " + class_names[np.argmax(prediction)]

            st.success(string)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
