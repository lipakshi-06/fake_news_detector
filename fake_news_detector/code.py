# ===============================
# FAKE NEWS DETECTOR PROJECT
# ===============================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Download Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Step 3: Load Dataset
df = pd.read_csv("news.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# Step 4: Data Cleaning Function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))   # remove symbols
    text = text.lower()                          # lowercase
    words = text.split()                         # tokenize
    words = [word for word in words if word not in stop_words]  # remove stopwords
    return " ".join(words)

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

print("\nText Cleaning Done!")

# Step 5: Define Features and Labels
X = df['text']
y = df['label']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Split Completed!")

# Step 7: Convert Text to Numbers using TF-IDF
vectorizer = TfidfVectorizer(max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("\nText Vectorization Completed!")

# Step 8: Train Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Step 9: Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# Step 11: Predict Custom News
# ===============================

def predict_news(news):
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    
    if prediction[0] == 1:
        return "REAL NEWS ✅"
    else:
        return "FAKE NEWS ❌"

# Example Test
print("\nCustom Prediction:")
sample = "Breaking: Government launches new education policy today"
print(sample)
print(predict_news(sample))

# ===============================
# END OF PROJECT
# ===============================