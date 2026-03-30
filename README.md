# 📰 Fake News Detector

## 📌 Overview
The Fake News Detector is a Machine Learning project that classifies news text as **Real** or **Fake** using Natural Language Processing (NLP).

It analyzes patterns in textual data rather than verifying facts from external sources, making it a fast and efficient classification system.

---

## 🎯 Objectives
- Detect fake news using machine learning
- Apply NLP techniques for text processing
- Build a command-line executable project
- Provide an optional graphical interface

---

## 🚀 Features
- Classifies news as REAL or FAKE
- Uses TF-IDF for feature extraction
- Logistic Regression model
- Command Line Interface (CLI) ✅ (Required)
- Streamlit GUI (Optional)

---

## 📂 Project Structure
fake_news_detector/
│
├── main.py # CLI version (runs in terminal)
├── app.py # Streamlit GUI (optional)
├── news.csv # Dataset
└── README.md

## ▶️ How to Run

### 🔹 Run CLI Version (Recommended & Required)
python main.py
👉 This runs the project in the terminal without any GUI.

---
### 🔹 Run GUI Version (Optional)
streamlit run app.py
👉 Opens a web interface in your browser.

---
## 🧪 Example Usage
### Input:Government bans mobile phones in schools
### Output:
Prediction: REAL NEWS ✅


---

## 🧠 How It Works

1. Text is cleaned (removes symbols and stopwords)
2. TF-IDF converts text into numerical features
3. Logistic Regression model analyzes patterns
4. Output is classified as Real or Fake

---

## ⚠️ Limitations

- Does not verify real-world facts
- Works best with news-style statements
- May give incorrect results for random or unrelated text

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit

---

## 📚 Learning Outcomes

- Understanding of NLP and text preprocessing
- Implementation of machine learning models
- Experience in building a real-world AI project
- Knowledge of CLI and GUI-based applications

---

## 👨‍💻 Author

Lipakshi Shah

---
## ⭐ Note
This project includes both:
- A **Command Line Interface (CLI)** to meet execution requirements  
- A **Graphical User Interface (GUI)** for better user experience  
