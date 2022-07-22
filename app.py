import streamlit as st
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import imutils
from PIL import Image
import numpy as np
import pyautogui
import os

os.environ['DISPLAY'] = ':0'


st.set_page_config(
    page_title="Image Similarity Detector",
    layout="wide",
    menu_items={
        'About': "This app was made by Rohit Arun."
    }
)

st.title('Image Similarity Detector')

def uploadImage(img_number):
    uploaded_file = st.file_uploader(f'Choose {img_number.lower()} image: ', type=(['png', 'jpeg', 'jpg']))
    if uploaded_file is not None:
        image = imutils.resize(np.asarray(Image.open(uploaded_file)), height=250)
        st.image(image, caption=f'{img_number} image')
        st.success(f'{img_number} image uploaded')
        image_object = [image, uploaded_file.name]
        return image_object
    else:
        pass

def img_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return ssim(img1, img2, full=True)

def img_mse(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return mse(img1, img2)
    
img_number_list = ['First', 'Second']

firstImageObject = uploadImage(img_number_list[0])
secondImageObject = uploadImage(img_number_list[1])

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True

if st.button('Calculate', on_click=callback) or st.session_state.button_clicked:
    if firstImageObject is None or secondImageObject is None:
        pass
    else:
        if firstImageObject[0].shape != secondImageObject[0].shape:
                st.error('The input images have different dimensions.')
                st.write('Would you like to resize one of the images corresponding to the other?')
                col_resize1, col_resize2 = st.columns(2)
                with col_resize1:
                    if st.button('Resize First Image'):
                        (H1, W1, _) = secondImageObject[0].shape
                        firstImageObject[0] = cv2.resize(firstImageObject[0], dsize = (W1, H1), interpolation=cv2.INTER_CUBIC)
                        st.image(firstImageObject[0], caption=f'{firstImageObject[1]}_{W1}x{H1}')
                with col_resize2:
                    if st.button('Resize Second Image'):
                        (H2, W2, _) = firstImageObject[0].shape
                        secondImageObject[0] = cv2.resize(secondImageObject[0], dsize = (W2, H2), interpolation=cv2.INTER_CUBIC)
                        st.image(secondImageObject[0], caption=f'{secondImageObject[1]}_{W2}x{H2}')
        else:
            pass

    if firstImageObject is None or secondImageObject is None:
        pass
    else:
        if firstImageObject[0].shape != secondImageObject[0].shape:
            pass
        else:
            (score_ssim, diff) = img_ssim(firstImageObject[0], secondImageObject[0])
            score_mse = img_mse(firstImageObject[0], secondImageObject[0])

            diff = (diff * 255).astype("uint8")
            thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(firstImageObject[0], (x, y), (x + w, y + h), (255, 20, 0), 2)
                cv2.rectangle(secondImageObject[0], (x, y), (x + w, y + h), (255, 20, 0), 2)

            tab1, tab2 = st.tabs(['Differences', 'Similarity Score'])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(firstImageObject[0], caption=f'{firstImageObject[1]}')
                with col2:        
                    st.image(secondImageObject[0], caption=f'{secondImageObject[1]}')

                col3, col4 = st.columns(2)
                with col3:
                    st.image(diff, caption='Difference')
                with col4:        
                    st.image(thresh, caption='Threshold')

            with tab2:
                col_ssim, col_mse = st.columns(2)
                with col_ssim:
                    score_ssim = round(score_ssim * 100, 2)
                    st.metric(label='SSIM', value=f'{score_ssim} %')
                with col_mse:
                    score_mse = round(score_mse, 2)
                    st.metric(label='MSE', value=f'{score_mse}')

    if st.button('Reset'):
        pyautogui.hotkey('ctrl', 'F5')

with st.expander("What is SSIM?"):
    st.write("""
        Structural Similarity Index Measure (SSIM) finds the perceived difference between two images based
        on their structural information. Unlike Mean Squared Error (MSE), it looks at corresponding local groups of pixels to find structural differences. These are calculated on the basis of luminosity, contrast and structure.
    """)

with st.expander("What is MSE?"):
    st.write("""
        Mean Sqaured Error (MSE) is calculated by taking the mean of the square of the difference between the two images, based on the differences in the intensities of corresponding pixels in both images. The larger the value of MSE, the greater is the difference between both images.
    """)