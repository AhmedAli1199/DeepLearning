import streamlit as st
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

@st.cache_resource
def load_sentiment_model():
    return tf.saved_model.load('movie_review_model')

try:
    loaded_model = load_sentiment_model()
    model_load_state = "Model loaded successfully!"
except Exception as e:
    loaded_model = None
    model_load_state = f"Error loading model: {str(e)}"


word_index = imdb.get_word_index()
reversed_word_index = {value: key for (key, value) in word_index.items()}

def preprocess(text):
    text = text.lower()
    words = text.split()
    encoded_review = [word_index.get(word, 0)+3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    # Convert to float32 tensor
    return tf.cast(padded_review, tf.float32)

def make_prediction(model, review):
    processed_review = preprocess(review)
    
    prediction = model.signatures["serving_default"](tf.constant(processed_review))
    prediction_value = list(prediction.values())[0].numpy()[0][0]
    
    sentiment = "Positive" if prediction_value > 0.5 else "Negative"
    return sentiment, prediction_value


st.title("Welcome to Movie Review Sentiment Analysis")
st.write("This is a simple web app to analyze the sentiment of movie reviews.")
st.write(model_load_state)  # Show if model loaded successfully

st.write("Please enter your movie review below:")
review = st.text_area("Movie Review", "Enter your review here...")

sentiment, probability = None, None
if st.button("Analyze"):
    if loaded_model is not None:
        sentiment, probability = make_prediction(loaded_model, review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Probability: {probability:.2f}")
    else:
        st.error("Model failed to load. Cannot analyze review.")
else:
    st.write("Click the button to analyze the sentiment of your review.")
    st.write("The model will predict whether the review is positive or negative based on the text you provide.")
st.write("Note: The model is trained on the IMDB dataset and may not perform well on reviews outside of that domain.")