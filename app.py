import streamlit as st
import cv2
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
import imutils
import numpy as np

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
    if uploaded_file is None:
        pass
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        st.image(img, caption=f'{uploaded_file.name} - {img.shape[1]}x{img.shape[0]}')
        resized_img = imutils.resize(img, height=300)
        gray_resized_image = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
        st.success(f'{img_number} image uploaded')
        image_object = [gray_resized_image, resized_img, uploaded_file.name]
        return image_object
    
img_number_list = ['First', 'Second']

first_image_obj, second_image_obj = uploadImage(img_number_list[0]), uploadImage(img_number_list[1]) 

if first_image_obj is None or second_image_obj is None:
    st.caption('Select two images having the same dimensions')
else:
    if first_image_obj[0].shape != second_image_obj[0].shape:
        st.error('The input images have different dimensions.')
        st.write('To calculate the similarity scores, the two images must have the same dimensions.')
    else:
        (score_ssim, diff) = ssim(first_image_obj[0], second_image_obj[0], full=True)
        score_mse = mse(first_image_obj[0], second_image_obj[0])

        diff = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if (cv2.contourArea(c)) > 100:
                cv2.rectangle(first_image_obj[1], (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(second_image_obj[1], (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                pass

        tab1, tab2 = st.tabs(['Differences', 'Similarity Score'])

        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(first_image_obj[1], caption=f'{first_image_obj[2]}')
            with col2:        
                st.image(second_image_obj[1], caption=f'{second_image_obj[2]}')
            # col3, col4 = st.columns(2)
            with col3:
                st.image(diff, caption='Difference')
            with col4:        
                st.image(thresh, caption='Threshold')

        with tab2:
            col_ssim, col_mse = st.columns(2)
            with col_ssim:
                score_ssim = round(score_ssim * 100, 2)
                st.metric(label='SSIM', value=f'{score_ssim} %')
                with st.expander("What is SSIM?"):
                    st.write("""
                        Structural Similarity Index Measure (SSIM) finds the perceived difference between two images based on their structural information. Unlike Mean Squared Error (MSE), it looks at corresponding local groups of pixels to find structural differences. These are calculated on the basis of luminosity, contrast and structure.
                    """)
            with col_mse:
                score_mse = round(score_mse, 2)
                st.metric(label='MSE', value=f'{score_mse}')
                with st.expander("What is MSE?"):
                    st.write("""
                        Mean Sqaured Error (MSE) is calculated by taking the mean of the square of the difference between the two images, based on the differences in the intensities of corresponding pixels in both images. The larger the value of MSE, the greater is the difference between both images.
                    """)
