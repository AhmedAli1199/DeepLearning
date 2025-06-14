import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model # To load the trained model
from tensorflow.keras.preprocessing.sequence import pad_sequences # To pad input sequences


model = load_model('lstm_next_word.h5')  # Load the trained model

# Load the tokenizer
with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(input_text,max_sequence_length=50):
    # Preprocess the input text
    input_text = input_text.lower()
    token_list = tokenizer.texts_to_sequences([input_text])[0]

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')

    # Make the prediction
    predicted = model.predict(token_list, verbose=0)

    # Get the index of the highest probability word
    predicted_index = np.argmax(predicted, axis=-1)[0]

    # Convert the index back to a word
    predicted_word = tokenizer.index_word[predicted_index]

    return predicted_word


# APP

st.title("Next Word Prediction App Using LSTM")

input = st.text_input("Enter a sentence:", "")
if input:
    next_word = predict_next_word(input, max_sequence_length=model.input_shape[1]+1) # USing max sequence length from the model
    st.write(f"The next word is: **{next_word}**")
# Add a footer
st.markdown("---")
st.markdown("Made with ❤️ by AHMED ALI")