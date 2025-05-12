# Run with: streamlit run app.py

import streamlit as st
from predict import classify_message

st.title("📬 Spam Email Classifier")
msg = st.text_area("Enter your message:")
if st.button("Predict"):
    st.write("Prediction:", classify_message(msg))
