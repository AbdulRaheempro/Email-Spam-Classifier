import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import requests

# Download resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def texttransform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words("english") and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# Load vectorizer & model
tfidf = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Page config
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

# Custom CSS (dark + neon + bigger title + footer)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #000000, #1a1a40, #0f3460);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    .big-title {
        font-size: 65px;  /* Increased from 55px */
        font-weight: bold;
        text-align: center;
        color: #00d9ff;
        text-shadow: 0px 0px 25px #00d9ff;
        margin-bottom: 10px;
    }
    textarea {
        border-radius: 15px !important;
        border: 2px solid #00d9ff !important;
        box-shadow: 0px 0px 10px #00d9ff !important;
        background-color: #0f2027 !important;
        color: #ffffff !important;
    }
    .stButton>button {
        background: #00d9ff;
        color: black;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0px 0px 15px #00d9ff;
    }
    .stButton>button:hover {
        background: #0077b6;
        color: white;
        transform: scale(1.05);
        box-shadow: 0px 0px 25px #00d9ff;
    }
    .result-box {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }
    .spam {
        background-color: rgba(255, 0, 0, 0.2);
        border: 2px solid #ff4d4d;
        color: #ff4d4d;
        text-shadow: 0px 0px 10px #ff4d4d;
        box-shadow: 0px 0px 20px #ff0000;
    }
    .ham {
        background-color: rgba(0, 255, 128, 0.2);
        border: 2px solid #00ff88;
        color: #00ff88;
        text-shadow: 0px 0px 10px #00ff88;
        box-shadow: 0px 0px 20px #00ff88;
    }
    .footer {
        text-align: center;
        font-size: 16px;
        color: #888888;
        margin-top: 20px;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">📧 Email Spam Classifier</p>', unsafe_allow_html=True)

# Lottie animation
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

animation = load_lottie("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(animation, height=200, key="email")

# Input box
input_sms = st.text_area("✍️ Enter your email text here:", height=150)

# Predict button
if st.button("🚀 Predict"):
    if input_sms.strip() != "":
        transform_sms = texttransform(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result-box spam">⚠️ This Email is SPAM ⚠️</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box ham">✅ This Email is NOT SPAM ✅</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text first!")

# Footer
st.markdown('<p class="footer">Made by ABDUL RAHEEM</p>', unsafe_allow_html=True)
