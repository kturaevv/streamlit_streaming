import os

import numpy as np
import cv2
import tempfile
import streamlit as st
from PIL import Image

from detection_pipeline.detect import run

# st.set_page_config(layout="wide", page_title="Image Upload", page_icon="🧊")
st.title("Grayscale Image / Video Converter")

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
file_inputs = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "mp4", "heic"], accept_multiple_files=True
)

if file_inputs is not None:
    for input_file in file_inputs:

        if input_file.name.split(".")[-1] == "mp4":

            temp_file_name = "temp_video_file.mp4"
            # Save the video file to disk
            with open(temp_file_name, "wb") as f:
                f.write(input_file.getbuffer())

            # Convert the video to grayscale
            processed_img = run(temp_file_name)

            # # Display the grayscale video
            st.video(open(processed_img, "rb").read())

            # # Remove the temporary video files
            os.remove(temp_file_name)
            os.remove(processed_img)

        else:
            # Open and display the original image
            image = Image.open(input_file)
            # Define tranform
            # transform = torchvision.transforms.Grayscale()
            # Convert to grayscale
            # processed_img = tranform(image)

            # Convert image to numpy array for processing
            image_tensor = np.array(image)
            # Run image through the model

            processed_img = run(image_tensor)

            # Depict tranformed image
            st.image(processed_img, caption="Processed Image", use_column_width=True)
