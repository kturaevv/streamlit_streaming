import streamlit as st
from PIL import Image


def grayscale(image):
    return image.convert('L')


st.title("Grayscale Image Converter")

# Upload an image
image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    # Open and display the original image
    image = Image.open(image_file)
    # Convert to grayscale
    grayscale_image = grayscale(image)
    st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
