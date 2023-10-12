import base64
from PIL import Image
import io
from predictor import predict_image_class, class_recall, class2_recall
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
import streamlit as st

# Cell detection model files
model_whole_name = 'vgg19_whole_slide.h5'
model_cell_name = 'vgg19_single_cell.h5'

def reset_prediction():
    if 'prediction' in st.session_state:
        del st.session_state['prediction']
        del st.session_state['confidence']
        del st.session_state['heatmap']
 
def predict_whole_slide(file_to_upload, gradcam, alpha_value):

     # Initialization    
    global model_whole_name
    if 'model_whole' not in st.session_state:
        print('Loading model...')
        st.session_state['model_whole'] = load_model(model_whole_name)
    st.header("Whole slide image")
    
    # Load the image and preprocess
    image = Image.open(file_to_upload)
    image_resized = image.resize((256,192), resample=Image.Resampling.BILINEAR)
    img_array = img_to_array(image_resized)
    img_array = preprocess_input_vgg19(img_array)

    # Inference (only once after loading an image)
    if 'prediction' not in st.session_state:
        st.session_state['prediction'], st.session_state['confidence'], st.session_state['heatmap'] = predict_image_class(
                                            model=st.session_state['model_whole'],
                                            image=img_array,
                                            image_type='Whole slide',
                                            gradcam_map=True)

    # Convert confidence to percentual string 
    confidence_string = str(round(st.session_state['confidence'] * 100, 2))
    # Convert heatmap to pillow:
    heatmap_pil = Image.fromarray(st.session_state['heatmap'])
    # Display prediction result:
    prediction_text = 'Prediction: ' + st.session_state['prediction'] + ' with probability ' + confidence_string + '%'
    st.subheader(prediction_text)

    # Overlay GradCAM++ heatmap to original image
    image_blended = Image.blend(image, heatmap_pil.resize(image.size), alpha_value)
    if gradcam:
        image_to_show = image_blended
    else:
        image_to_show = image
    
    st.image(image_to_show, use_column_width="auto")
 
def predict_single_cell(file_to_upload, gradcam, alpha_value):

    # Initialization
    global model_cell_name
    if 'model_cell' not in st.session_state:
        print('Loading model...')
        st.session_state['model_cell'] = load_model(model_cell_name, custom_objects={"class2_recall": class2_recall})
    st.header("Single cell image")

    # Load the image and preprocess
    image = Image.open(file_to_upload)
    image_resized = image.resize((128,128), resample=Image.Resampling.BILINEAR)
    img_array = img_to_array(image_resized)
    img_array = preprocess_input_vgg19(img_array)

    # Inference (only once per image)
    if 'prediction' not in st.session_state:
        st.session_state['prediction'], st.session_state['confidence'], st.session_state['heatmap'] = predict_image_class(
                                            model=st.session_state['model_cell'],
                                            image=img_array,
                                            image_type='Single cell',
                                            gradcam_map=True)

    # Convert confidence to percentual string
    confidence_string = str(round(st.session_state['confidence'] * 100, 2))
    # Convert heatmap to pillow and then base64 format
    heatmap_pil = Image.fromarray(st.session_state['heatmap'])
    # Display prediction result
    prediction_text = 'Prediction: ' + st.session_state['prediction'] + ' with probability ' + confidence_string + '%'
    st.header(prediction_text)

    # Overlay GradCAM++ heatmap to original image
    image_blended = Image.blend(image, heatmap_pil.resize(image.size), alpha_value)
    if gradcam:
        image_to_show = image_blended
    else:
        image_to_show = image

    st.image(image_to_show, use_column_width="auto")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cervical cancer cell detection", initial_sidebar_state="expanded")
    st.title('Cervical cancer cell detection')
    st.info('''Please select either a whole slide or a single cell image.
            GradCAM++ shows the image regions relevant for the recognition.''')
    
    with st.sidebar:
        st.title("Upload a microscope image")
        type_of_image = st.radio('Type of image:', ['Whole slide', 'Single cell'])
        file_to_upload = st.file_uploader("Choose a file:", on_change=reset_prediction)
        gradcam = st.checkbox('GradCAM++')
        alpha_value = st.slider('Heatmap overlay:', min_value=0.0, max_value=1.0, value=0.35)
    
    if file_to_upload is not None:
        if type_of_image == "Whole slide":
            predict_whole_slide(file_to_upload, gradcam, alpha_value)
        elif type_of_image == "Single cell":
            predict_single_cell(file_to_upload, gradcam, alpha_value)
