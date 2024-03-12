
import streamlit as st
import cv2
import numpy as np
import joblib

# Set page title
st.set_page_config(layout="wide", page_title="Water Analysis App")
# Add custom CSS to remove space above the title
custom_css = """
<style>
.stApp {
    margin-top: 0;
    padding-top: 0;
    
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Set the page title
#st.set_page_config(layout="wide", page_title="Water Analysis App")

# Create a centered title using HTML and CSS within Markdown
centered_title = """
<div style="text-align: center;">
    <h1>Water Quality Analysis using Satellite Images Model</h1>
</div>
"""


st.markdown(centered_title, unsafe_allow_html=True)
    # Load other models
model_path1 = r"ph.pkl"
model_path2 = r"ag.pkl"
model_path3 = r"adm.pkl"
model_path4 = r"a.pkl"
model_path5 = r"bbdm.pkl"
model_path6 = r"bbch.pkl"
model_path7 = r"bb.pkl"

model1 = joblib.load(model_path1)
model2 = joblib.load(model_path2)
model3 = joblib.load(model_path3)
model4 = joblib.load(model_path4)
model5 = joblib.load(model_path5)
model6 = joblib.load(model_path6)
model7 = joblib.load(model_path7)

# Function to load image
import io

def load_image(uploaded_file):
    # Convert the uploaded file to bytes
    content = uploaded_file.getvalue()
    # Read the image from bytes
    img = cv2.imdecode(np.frombuffer(content, np.uint8), 1)
    return img

# Function to find mode pixel
def run(img):
    pixel_counts = {}
    for row in img:
        for pixel in row:
            pixel_tuple = tuple(pixel)
            pixel_counts[pixel_tuple] = pixel_counts.get(pixel_tuple, 0) + 1

    mode_pixels = []
    max_count = max(pixel_counts.values())
    for pixel, count in pixel_counts.items():
        if count == max_count:
            mode_pixels.append(pixel)

    return mode_pixels[0]

# Function to calculate predicted wavelength
def calculate_wavelength(rgb):
    Lab_OpenCV = [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]
    predicted_wavelength = model.predict([Lab_OpenCV])
    return predicted_wavelength[0]

# Load the trained model
model = joblib.load('wavelength_model.pkl')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    #st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    # Read the image file
    image = load_image(uploaded_file)

    # Calculate the mode of pixels
    mode_pixel = run(image)

    # Calculate predicted wavelength for the mode pixel
    predicted_wavelength = calculate_wavelength(mode_pixel)
    new_wavelength = predicted_wavelength

    ph = model1.predict([[new_wavelength]])[0]
    ag = model2.predict([[new_wavelength]])[0]
    adm = model3.predict([[new_wavelength]])[0]
    a = model4.predict([[new_wavelength]])[0]
    bbdm = model5.predict([[new_wavelength]])[0]
    bbch = model6.predict([[new_wavelength]])[0]
    bb = model7.predict([[new_wavelength]])[0]
    # Display the predicted wavelength
    #st.write("Predicted Wavelength:", predicted_wavelength)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", width=400)
        st.write("Predicted Wavelength:", predicted_wavelength)
        




    # Display predicted values
    with col2:
        st.write("Predicted values for wavelength", new_wavelength, ":")
        st.write("pH:", ph)
        st.write("Ag:", ag)
        st.write("Adm:", adm)
        st.write("A:", a)
        st.write("Bbdm:", bbdm)
        st.write("Bbch:", bbch)
        st.write("Bb:", bb)
