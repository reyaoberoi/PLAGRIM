import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
import random
from googleapiclient.discovery import build

#GOOGLE API KEYS
API_KEY = "AIzaSyDHpnJQWqLrJJEqdu5La_win4wqDM8KZ8I"
SEARCH_ENGINE_ID = "f195b85d860c74e5a"

#Configuration and Initialization
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
MODEL_FILE = 'ai_detector_model.pkl'

st.set_page_config(page_title=" PLAGRIM : Plaigiarism & AI Content Checker", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .main-header { font-size: 2.5em; font-weight: 700; color: #1e3a8a; text-align: center; margin-bottom: 20px; }
    .subheader { font-size: 1.2em; color: #4b5563; text-align: center; margin-bottom: 30px; }
    .stButton>button {
        background-color: #1d4ed8;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: background-color 0.3s;
        width: 100%;
    }
    .stButton>button:hover { background-color: #2563eb; }
    .result-card { padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 20px; }
    .plagiarism-flag { color: #b91c1c; font-weight: 600; }
    .ai-flag { color: #059669; font-weight: 600; }
    .sidebar-header { font-weight: bold; font-size: 1.1em; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

#1. Model Loading
@st.cache_resource
def load_ai_model_assets():
    try:
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = joblib.load(f)
        with open(MODEL_FILE, 'rb') as f:
            model = joblib.load(f)
        st.sidebar.success("AI Model loaded successfully.")
        return vectorizer, model
    except FileNotFoundError:
        st.sidebar.error("Error: Model file(s) not found.")
        return None, None

def predict_ai_probability(text, vectorizer, model):
    if vectorizer is None or model is None:
        return 0.5, "Model Load Error"
    text_transformed = vectorizer.transform([text])
    probabilities = model.predict_proba(text_transformed)[0]
    ai_prob = probabilities[1]
    prediction = "AI-Generated" if ai_prob > 0.5 else "Human-Written"
    return ai_prob, prediction

#2. REAL WEB PLAGIARISM CHECK
def split_text_into_shingles(text, min_length=15):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:)\s', text.replace('\n', ' '))
    shingles = [s.strip() for s in sentences if len(s.split()) >= min_length]
    return shingles[:8]

def check_plagiarism_via_search(text):
    st.info("Initiating Real Web Plagiarism Check...")

    shingles = split_text_into_shingles(text)
    if not shingles:
        return [], "Text is too short for a web check."

    service = build("customsearch", "v1", developerKey=API_KEY)
    results = []

    for shingle in shingles:
        query = f'"{shingle}"'
        try:
            response = service.cse().list(q=query, cx=SEARCH_ENGINE_ID, num=3).execute()
            if "items" in response:
                for item in response["items"]:
                    results.append({
                        "phrase": shingle,
                        "title": item.get("title"),
                        "uri": item.get("link")
                    })
        except Exception as e:
            return [], f"Search error: {e}"

    if not results:
        return [], f"Checked {len(shingles)} key phrases. No plagiarism found."

    return results, f"Plagiarism found in {len(results)} sentence(s)."

#MAIN APP UI
def main():
    st.markdown('<h1 class="main-header">📄PLAGRIM: Plagiarism & AI Content Checker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Powered by ML and Real-Time Web Search.</p>', unsafe_allow_html=True)

    st.sidebar.markdown('<p class="sidebar-header">Model Status</p>', unsafe_allow_html=True)
    vectorizer, model = load_ai_model_assets()

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area("Paste Text for Analysis:", height=300)

    with col2:
        uploaded_file = st.file_uploader("Upload .txt Document:", type=['txt'])
        if uploaded_file is not None:
            text_input = uploaded_file.read().decode("utf-8")
            st.success("File uploaded and content extracted.")

        run_button = st.button("Run Comprehensive Check", type="primary")

    if run_button and text_input:
        clean_text = text_input.strip()

        if len(clean_text) < 50:
            st.warning("Enter at least 50 characters for analysis.")
            return

        with st.spinner("Analyzing content..."):

            # AI CONTENT CHECK (With your required conditions)
            ai_prob, ai_prediction = predict_ai_probability(clean_text, vectorizer, model)

            st.markdown('<div class="result-card" style="background-color: #e0f2f1;">', unsafe_allow_html=True)
            st.subheader("🤖 AI Content Detection Results")

            ai_score = ai_prob * 100
            st.progress(ai_prob, text=f"AI Probability: {ai_score:.2f}%")

            if ai_score >= 70:
                st.markdown(
                    f"**Result:** <span class='plagiarism-flag'>**High Likelihood of AI-Generated Content**</span>",
                    unsafe_allow_html=True
                )
            elif ai_score >= 40:
                st.markdown(
                    f"**Result:** Medium Likelihood (Review Needed)",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"**Result:** <span class='ai-flag'>**Likely Human-Written**</span>",
                    unsafe_allow_html=True
                )

            st.markdown(f"*(Classification: {ai_prediction})*", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

            #WEB PLAG CHECK
            results, message = check_plagiarism_via_search(clean_text)

            st.markdown('<div class="result-card" style="background-color: #ffe0e0;">', unsafe_allow_html=True)
            st.subheader("🔗 Web Plagiarism Check Results")
            st.info(message)

            if results:
                st.error(f"🚨 Plagiarism Detected! ({len(results)} matches)")
                for item in results:
                    st.markdown("---")
                    st.markdown(f"**Plagiarized Phrase:** <span class='plagiarism-flag'>*{item['phrase']}*</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"**Source:** <a href='{item['uri']}' target='_blank' style='text-decoration:none;color:#1d4ed8;'>{item['title']}</a>",
                        unsafe_allow_html=True
                    )
            else:
                st.success("✅ No Direct Plagiarism Found")

            st.markdown('</div>', unsafe_allow_html=True)

    elif run_button:
        st.warning("Paste text or upload a file.")

if __name__ == "__main__":
    main()
