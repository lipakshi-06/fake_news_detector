import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

print("Loading dataset...")

df = pd.read_csv("news.csv")
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(max_df=0.7)
X = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X, y)

print("Model trained successfully!")

def predict_news(news):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return "REAL NEWS ✅" if prediction[0] == 1 else "FAKE NEWS ❌"

print("\n📰 Fake News Detector (CLI)")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter news text: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    if user_input.strip() == "":
        print("⚠️ Enter something\n")
        continue

    result = predict_news(user_input)
    print("Prediction:", result, "\n")