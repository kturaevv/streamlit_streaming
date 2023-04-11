import streamlit as st
from PIL import Image

import cv2
import tempfile
import os

# Function to convert a video to grayscale using OpenCV
def convert_to_grayscale(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video metadata
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to write the grayscale frames to
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", dir=".", delete=False)
    out = cv2.VideoWriter(
        temp_file.name,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (frame_width, frame_height),
        False,
    )

    # Loop through each frame in the video and convert it to grayscale
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        out.write(gray_frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Return the path to the grayscale video file
    return temp_file.name
