# Imports
import streamlit as st
from tensorflow import keras
import pandas as pd
import numpy as np
from pathlib import Path

import time
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from gtts import gTTS
from IPython.display import Audio

#caption_model_vgg = keras.models.load_model('../models/caption_mode_with_vgg16.h5')
model_path = 'models/caption_mode_with_vgg16.h5'
caption_model_vgg = keras.models.load_model(model_path)

#df = pd.read_csv('../data/cleaned_caption.csv')
csv_file_path = 'data/cleaned_caption.csv'
df = pd.read_csv(csv_file_path)

# Title, head and 
st.title(":blue[Every Photo Has A Story To Tell]")
st.header(':blue[Do you want to have a caption of your image?]')
st.write("It's pretty simple!")

# Request for upload afile
image = st.file_uploader("Just upload your image here")
if image is not None:

    # Define spinner
    with st.spinner('Wait for it...'):
        time.sleep(5)

    # Instantiate VGG16() model for image feature extraction
    vgg16_model = VGG16()

    # Restructure model
    vgg16_model = Model(inputs = vgg16_model.inputs , outputs = vgg16_model.layers[-2].output)

    # Load image
    img_size= 224

    # image= image path
    img = load_img(image,target_size = (img_size,img_size))
    
    # Convert image to array
    img = np.array(img)
    #img = img_to_array(img)
        
    # Reshape the image by adding another dimension to preprocess in a RGB
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    #img = np.expand_dims(img, axis=0)
    
    # preprocess image for scaling the pixel values
    img = preprocess_input(img)

    #img = img/255
    #img = np.expand_dims(img, axis=0)

    # Extract features
    img_feat = vgg16_model.predict(img, verbose=0)

    # Get list of captions
    captions = df['caption'].tolist()

    # Create tokenizer and apply on captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)    

    # Determine maximum caption length based on the number of words ditribution in 01_EDA_and_Cleaning
    max_caption_length = 25

    # Function for finding words based on the tokenized captions
    def idx_to_word(integer,tokenizer):
    
        for word, index in tokenizer.word_index.items():
            if index==integer:
                return word
        return None

    gen_caption = 'startsen'
    for i in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([gen_caption])[0]       # Convert the text data to a numerical format that can be fed into embedding layer
        sequence = pad_sequences([sequence], max_caption_length)        # Pad sequences to a specified length (max_length)

        y_pred = caption_model_vgg.predict([img_feat,sequence])
        y_pred = np.argmax(y_pred)                                      # Find the index of the class with the highest probability in the
                                                                        # probability distribution over the classes after softmax() function in our model
        word = idx_to_word(y_pred, tokenizer)                           # Use idx_to_word function for generating the word
        
        if word is None:
            break
            
        gen_caption += ' ' + word                                       # Add generated word to the caption
        
        if word == 'endsen':
            break
    
    
    # Remove start and end tags from generated caption
    gen_caption = ' '.join([cap for cap in gen_caption.split() if cap not in ['startsen', 'endsen']])

    # End of spinner
    st.success('Done!')

    # Subheader for loaded image
    st.subheader(':blue[Here is your image]')

    # Show the image
    st.image(image)

    # Subheader for generated caption
    st.subheader(':blue[And, here is the caption]')

    # Show the generated caption
    st.write(gen_caption)

    # Subheader for speech
    st.subheader(':blue[No time to read? No problem, click here!]')

    # Generate the speech from caption
    tts = gTTS(gen_caption)
    tts.save('1.wav')
    sound_file = '1.wav'
    #Audio(sound_file, autoplay=True)
    st.audio(sound_file)

    






