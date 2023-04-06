import streamlit as st
from PIL import Image
import torchvision

# st.set_page_config(layout="wide", page_title="Image Upload", page_icon="🧊")
st.title("Grayscale Image Converter")

# Disable scrolling and remove menu
css = """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            section.main > div:has(~ footer ) {
                padding-bottom: 5px;
            }
        </style>
        """
st.markdown(css, unsafe_allow_html=True)
# Upload an image
image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if image_file is not None:
    # Open and display the original image
    image = Image.open(image_file)
    # Define tranform
    transform = torchvision.transforms.Grayscale()
    # Convert to grayscale
    grayscale_image = transform(image)
    # Depict tranformed image
    st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)
