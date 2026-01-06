import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd
import random
from googleapiclient.discovery import build

# GOOGLE API KEYS
API_KEY = ""
SEARCH_ENGINE_ID = "f195b85d860c74e5a"

# Configuration
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
MODEL_FILE = 'ai_detector_model.pkl'

st.set_page_config(
    page_title="PLAGRIM : Plagiarism & AI Content Checker",
    layout="wide"
)

# ---------------- SQUARE CARD UI (ONLY CSS CHANGE) ----------------
st.markdown("""
<style>

/* Page background */
.stApp {
    background-color: #301934;
}

/* Centered square container */
.block-container {
    max-width: 800px;
    background-color: #BDB76B;
    padding: 32px 36px 40px 36px;
    border-radius: 18px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    margin-top: 40px;
    margin-bottom: 40px;
}

/* Header */
.main-header {
    font-size: 2.3em;
    font-weight: 700;
    color: #1e293b;
    text-align: center;
    margin-bottom: 8px;
}

.subheader {
    font-size: 1.05em;
    color: #008000;
    text-align: center;
    margin-bottom: 28px;
}

/* Text area */
textarea {
    aspect-ratio: 1 / 1;
    resize: none;
    max-width: 100%;
}
    border-radius: 6px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: bold !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 15px;
    border: 1px dashed #cbd5e1;
    padding: 12px;
    background-color: #ffffff;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: px;
    padding: 10px 16px;
    font-size: 0.95em;
    width: 100%;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Result cards */
.result-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 18px;
    margin-top: 20px;
    border: 1px solid #e5e7eb;
}

/* Flags */
.plagiarism-flag {
    color: #dc2626;
    font-weight: 600;
}

.ai-flag {
    color: #16a34a;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e5e7eb;
}

.sidebar-header {
    font-weight: 600;
    font-size: 1.05em;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_ai_model_assets():
    try:
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = joblib.load(f)
        with open(MODEL_FILE, 'rb') as f:
            model = joblib.load(f)
        st.sidebar.success("Model loaded successfully.")
        return vectorizer, model
    except FileNotFoundError:
        st.sidebar.error("Model file(s) not found.")
        return None, None

def predict_ai_probability(text, vectorizer, model):
    text_transformed = vectorizer.transform([text])
    probabilities = model.predict_proba(text_transformed)[0]
    ai_prob = probabilities[1]
    prediction = "AI-Generated" if ai_prob > 0.5 else "Human-Written"
    return ai_prob, prediction

# ---------------- PLAGIARISM CHECK ----------------
def split_text_into_shingles(text, min_length=15):
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\:)\s',
        text.replace('\n', ' ')
    )
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
        try:
            response = service.cse().list(
                q=f'"{shingle}"',
                cx=SEARCH_ENGINE_ID,
                num=3
            ).execute()
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

# ---------------- MAIN APP ----------------
def main():
    st.markdown(
        '<h1 class="main-header"> PLAGRIM: Plagiarism & AI Content Checker</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subheader">Powered by ML and Real-Time Web Search.</p>',
        unsafe_allow_html=True
    )

    st.sidebar.markdown(
        '<p class="sidebar-header">Model Status</p>',
        unsafe_allow_html=True
    )
    vectorizer, model = load_ai_model_assets()

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area("Paste Text for Analysis:", height=280)

    with col2:
        uploaded_file = st.file_uploader("Upload .txt Document:", type=['txt'])
        if uploaded_file:
            text_input = uploaded_file.read().decode("utf-8")
            st.success("File uploaded and content extracted.")

        run_button = st.button("Run Comprehensive Check")

    if run_button and text_input:
        clean_text = text_input.strip()

        if len(clean_text) < 50:
            st.warning("Enter at least 50 characters for analysis.")
            return

        with st.spinner("Analyzing content..."):
            ai_prob, ai_prediction = predict_ai_probability(
                clean_text, vectorizer, model
            )

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🤖 AI Content Detection Results")
            st.progress(ai_prob, text=f"AI Probability: {ai_prob*100:.2f}%")

            if ai_prob >= 0.7:
                st.markdown(
                    "<span class='plagiarism-flag'>High likelihood of AI-generated content</span>",
                    unsafe_allow_html=True
                )
            elif ai_prob >= 0.4:
                st.markdown("Medium likelihood (review needed)")
            else:
                st.markdown(
                    "<span class='ai-flag'>Likely human-written</span>",
                    unsafe_allow_html=True
                )

            st.markdown(f"*Classification: {ai_prediction}*")
            st.markdown('</div>', unsafe_allow_html=True)

            results, message = check_plagiarism_via_search(clean_text)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🔗 Web Plagiarism Check Results")
            st.info(message)

            if results:
                for item in results:
                    st.markdown("---")
                    st.markdown(
                        f"**Plagiarized Phrase:** "
                        f"<span class='plagiarism-flag'>{item['phrase']}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"[Source: {item['title']}]({item['uri']})"
                    )
            else:
                st.success("No direct plagiarism found.")

            st.markdown('</div>', unsafe_allow_html=True)

    elif run_button:
        st.warning("Paste text or upload a file.")

if __name__ == "__main__":
    main()
