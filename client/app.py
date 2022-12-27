import streamlit as st
import requests
from PIL import Image, ImageChops
import numpy as np
import cv2

# Preprocessing the image data
def trim(image):
    image=Image.fromarray(image)
    bg = Image.new(image.mode, image.size, image.getpixel((0,0))) # black background
    diff = ImageChops.difference(image, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return np.array(image.crop(bbox))
# cv2.INTER_AREA: examiner les pixels voisins et utiliser ces voisins pour augmenter ou diminuer optiquement la taille de l’image sans introduire de distorsions
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        #r = width / float(w)
        #dim = (width, int(h * r))
        dim = (width,height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
def cleaned_image(image):
    bg=np.ones((28,28))
    bg = bg*255
    image=trim(image)
    if image is not None:
        image = image_resize(image,height=28)
    else:
        return None
    #image=image_resize(image,height=28)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hh, ww = bg.shape
    h, w = image_binary.shape
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    if xoff<=0:
        image=image_resize(image,height=28,width=28)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, image_binary = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = image_binary.shape
        yoff = round((hh-h)/2)
        xoff = round((ww-w)/2)
    result = bg.copy()
    result[yoff:yoff+h, xoff:xoff+w] = image_binary
    return result


        
def run():
    st.title("Handwritten Character Recognition")
    # Allow the user to select an image file
    image_file = st.file_uploader("Choose an image", type=["jpg", "png"])
    c1, c2= st.columns(2)
    if image_file is not None:
        # Load the image data and convert it to a NumPy array
        image_data = image_file.read()
        # Decode the image data and convert it to a NumPy array
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Display the image
        #st.image(image)
        # Apply the image processing functions to the image
        cleaned_imaged = cleaned_image(image)
        vect = cleaned_imaged.flatten().tolist()
        c1.header('Original Image')
        c1.image(image)
        #c1.write(image.shape)
        
        if st.button("Predict"):
            response = requests.post("http://127.0.0.1:8000/predict", json={"image": vect})
            response_json = response.json()

            # Display the classification result
            c2.header('Output')
            c2.subheader('Predicted class :')
            c2.write(response_json['prediction'])
            #st.markdown(f"Prediction: {response_json['prediction']}")
    
if __name__ == '__main__':
    #by default it will run at 8501 port
    run()

