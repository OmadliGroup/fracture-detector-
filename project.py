import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
model = load_model('model.h5')

def main():
    st.title('Fracture Detection Web App')
    st.write('***NOTE:** This app predicts fracture type based on the specific open-source data used to train this model*')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read the image file
        img = Image.open(uploaded_file)
        
        # Resize the image to match the input shape of the model (200x200 pixels)
        img = img.resize((200, 200))
        
        # Convert image to numpy array
        img_array = np.array(img)
        
        # Ensure the image has three color channels (RGB)
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("Image does not have three color channels (RGB)")
            return
        
        # Normalize pixel values (optional, depending on your model)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction using your ML model
        result = model.predict(img_array)
        
        classes = ['Avulsion Fracture', 'Greenstick fracture', 'Hairline fracture', 
                   'Longitudinal fracture', 'Oblique Dislocation', 
                   'Pathological fracture', 'Spiral fracture']
        
        prediction = classes[np.argmax(result)]

        st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
