import streamlit as st
import pandas as pd
import os
import joblib
import string
import requests
import regex
import nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from streamlit_lottie import st_lottie


current_directory = os.path.dirname(__file__)
os.chdir(current_directory)
# Load trained model
svm_model = joblib.load(os.path.join(current_directory, 'svm_model.joblib'))

# Load TF-IDF vectorizer
vectorizer = joblib.load(os.path.join(current_directory, 'tfidf_vectorizer.joblib'))


def preprocess_text(text):
    # Normalisasi teks
    text = text.lower()

    # Menghilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = text.split()

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Menggabungkan kembali token menjadi teks
    processed_text = ' '.join(tokens)

    return processed_text

st.set_page_config(
    page_title="Classify",
    page_icon=":imp:",
    layout="wide"
)

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()
    

# Header
lottie_sad = load_lottieurl("https://lottie.host/5f336e56-49d4-4c3d-b5f6-8e5709b3ac69/0C4dF9eI74.json")
lottie_happy = load_lottieurl("https://lottie.host/d9228243-1b2a-4cb4-b33b-194f80208b1d/p5nk7fffft.json")

# Container
with st.container():
    cl1, cl2, cl3 = st.columns((1,1,1))
    with cl2:
        st.title('Depression Detection')
    st.write("---")

with st.container():
    cl1, cl2, cl3 = st.columns((1,2,1))
    with cl2:
        user_input = st.text_area('Enter your tweet:')
        run_button = st.button('Classify')

with st.container():
    cl1, cl2, cl3 = st.columns((2,2,2))
    with cl2:
        if user_input:
            # Preprocess the user input
            user_input = preprocess_text(user_input)

            # Transform the user input to the TF-IDF format
            user_input_tfidf = vectorizer.transform([user_input])

            # Predict the class of the user input
            prediction = svm_model.predict(user_input_tfidf)

            # Print the prediction
            if prediction == 1:
                st_lottie(lottie_sad, height=80, key='coding')
                st.error('The tweet above is classified as depressing')
            else:
                st_lottie(lottie_happy, height=80, key='coding')
                st.success('The tweet above is classified as non-depressing')