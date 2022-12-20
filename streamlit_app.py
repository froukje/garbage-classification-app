import io
import os
import numpy as np
import requests
import streamlit as st
from PIL import Image

#url = 'http://127.0.0.1:3000/predict' 
url = os.getenv('API_ENDPOINT', 'http://127.0.0.1:3000/predict')
#url = 'http://garbage_classification_service:nnarakeap2ra6wew:3000/predict'

# Create the header page content
st.title("Garbage Classification App")
st.markdown(
    "### Classify your garbage images between 'cardboard', 'glass', 'metal', 'paper', 'plastic' and 'trash'",
    unsafe_allow_html=True,
)

# Upload a cover image
with open("garbage.jpg", "rb") as f:
    st.image(f.read(), use_column_width=True)

st.text("Upload your image.")

def predict(img):
    """
    A function that sends a prediction request to the API and returns the gabage class.
    """

    # Convert the bytes image to a NumPy array
    bytes_image = img.getvalue()
    numpy_image_array = np.array(Image.open(io.BytesIO(bytes_image)))

    # Send the image to the API
    response = requests.post(
        url,
        headers={"content-type": "text/plain"},
        data=str(numpy_image_array),
    )

    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Status: {}".format(response.status_code))

# two image input components — one for file uploads and another for a web-cam input
def main():
    img_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if img_file is not None:

        with st.spinner("Predicting..."):
            prediction = predict(img_file)
            st.success(f"Your image is {prediction}")

    camera_input = st.camera_input("Or take a picture")
    if camera_input is not None:

        with st.spinner("Predicting..."):
            prediction = predict(camera_input)
            st.success(f"Your image is {prediction}")


if __name__ == "__main__":
    main()
