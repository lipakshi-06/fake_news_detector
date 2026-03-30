import streamlit as st
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords

# ------------------------------
# TEXT CLEANING
# ------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ------------------------------
# LOAD DATA + TRAIN MODEL (AUTO)
# ------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("news.csv")   # make sure file exists

    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_df=0.7)
    X = vectorizer.fit_transform(X)

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_model()

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
def predict_news(news):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "REAL NEWS ✅"
    else:
        return "FAKE NEWS ❌"

# ------------------------------
# UI
# ------------------------------
st.title("📰 Fake News Detector")

st.write("⚠️ Enter news-style sentence for best results")

user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_news(user_input)
        st.success(result)