import streamlit as st
from PIL import Image
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 
model = keras.models.load_model('comix.h5')

st.title("Face to comic")

def load_image(image_file):
	img = Image.open(image_file)
	return img

def doctor(image, size):
    image = image.resize((size, size))
    image_sequence = image.getdata()
    image_array = np.array(image_sequence)
    image_sequence = image_array
    image_sequence = np.resize(image_sequence, ((size, size, 3)))
    return (image_sequence/255)

st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

          # To See details
          #file_details = {"filename":image_file.name, "filetype":image_file.type,
           #   "filesize":image_file.size}
          #st.write(file_details)
          img = load_image(image_file)
          st.image(img,width=250)
          px = doctor(img, 100)
          px = np.resize(px, (1, 100, 100, 3))
          ppy = model.predict(px)
          st.image(ppy,width=250)
