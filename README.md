# 📧 Spam Email Classifier

A machine learning project to classify SMS/email messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a Naive Bayes classifier.

### 🔗 Try it Live
👉 [Click here to use the web app](https://spam-email-classifier-rpfgu7dozxfcd5telrsuyl.streamlit.app/)

---

## 🚀 Features

- ✅ Classifies messages as **Spam** or **Ham**
- ✅ Clean and intuitive web interface with **Streamlit**
- ✅ Preprocessing includes stopword removal, stemming, and tokenization
- ✅ Trained on the **SMS Spam Collection Dataset**
- ✅ Lightweight and fast — great for learning ML fundamentals

---

## 🧠 Model Overview

- **Algorithm**: Multinomial Naive Bayes
- **Vectorization**: TF-IDF with 3000 features
- **Accuracy**: ~95% on test data
- **Preprocessing**:
  - Lowercasing
  - Removing punctuation
  - Removing stopwords
  - Stemming using NLTK

---

## 🗂️ Project Structure
spam-email-classifier/
│
├── data/
│ └── spam.csv # SMS Spam dataset
│
├── model.pkl # Trained spam classifier
├── vectorizer.pkl # Saved TF-IDF vectorizer
│
├── app.py # Streamlit app
├── train_model.py # Training script
├── predict.py # Command-line predictor
├── evaluate_model.py # Model evaluation script
├── preprocessing.py # Text cleaning functions
│
├── requirements.txt # Python dependencies
└── README.md # You're here!

## 🛠️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/spam-email-classifier.git
   cd spam-email-classifier

    Create a virtual environment and activate it:

python -m venv .venv
.\.venv\Scripts\activate  # On Windows

Install dependencies:

pip install -r requirements.txt

Run the app:

    streamlit run app.py

📦 Dataset

    SMS Spam Collection Dataset (UCI)

📜 License

This project is open-source and free to use.
👤 Sudhanshu Agarwal

Made with 💻 and ☕ by Sudhanshu Agarwal.
