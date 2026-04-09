import streamlit as st
import pickle
import pandas as pd
import os
from preprocess import clean_text

# Page Config
st.set_page_config(page_title="AI Spam Filter", page_icon="📧")

# Load Paths
MODEL_PATH = 'models'
NB_MODEL_PATH = os.path.join(MODEL_PATH, 'naive_bayes_model.pkl')
LR_MODEL_PATH = os.path.join(MODEL_PATH, 'logistic_regression_model.pkl')
VEC_PATH = os.path.join(MODEL_PATH, 'vectorizer.pkl')

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox("Menu", ["Home", "About"])

# Load Models (Handle case where files aren't there yet)
try:
    with open(NB_MODEL_PATH, 'rb') as f:
        nb_model = pickle.load(f)
    with open(LR_MODEL_PATH, 'rb') as f:
        lr_model = pickle.load(f)
    with open(VEC_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    models_loaded = True
except FileNotFoundError:
    st.error("Models not found. Please run `python train_model.py` first.")
    models_loaded = False

# Main Dashboard
st.title("📧 AI Spam Email Detector")
st.markdown("""
This system uses **NLP** to classify messages as **Spam** or **Ham**. 
Choose your preferred algorithm below.
""")

# User Input
st.header("Input Message")
user_input = st.text_area("Enter your message here:", height=200)

col1, col2 = st.columns(2)

with col1:
    btn_nb = st.button("Predict (Naive Bayes)")
with col2:
    btn_lr = st.button("Predict (Logistic Reg.)")

if not models_loaded:
    st.warning("⚠️ Wait until you train the model.")
elif user_input and (btn_nb or btn_lr):
    if btn_nb:
        # Prediction
        processed_input = [clean_text(user_input)]
        features = vectorizer.transform(processed_input)
        prediction = nb_model.predict(features)[0]
        prob = nb_model.predict_proba(features)[0]
        
        # Display Result
        col_result_nb = st.columns([1, 3])
        with col_result_nb[0]:
            st.metric(label="Classification", value="Spam 😡" if prediction == 1 else "Safe ✅")
        with col_result_nb[1]:
            st.progress(prob[prediction])
            st.caption(f"Confidence: {prob[prediction]*100:.2f}%")
            
    elif btn_lr:
        processed_input = [clean_text(user_input)]
        features = vectorizer.transform(processed_input)
        prediction = lr_model.predict(features)[0]
        prob = lr_model.predict_proba(features)[0]
        
        # Display Result
        col_result_lr = st.columns([1, 3])
        with col_result_lr[0]:
            st.metric(label="Classification", value="Spam 😡" if prediction == 1 else "Safe ✅")
        with col_result_lr[1]:
            st.progress(prob[prediction])
            st.caption(f"Confidence: {prob[prediction]*100:.2f}%")

else:
    st.info("Please enter text above and click a button to start prediction.")

# Footer
st.divider()
st.write("**Note:** This is a demonstration model trained on a subset of the SMS Spam Collection dataset.")
