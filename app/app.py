import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import re
from summa.summarizer import summarize
import nltk

# --- Streamlit Config ---
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")

# Ensure NLTK tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# --- Helpers ---
def clean_text(text):
    return text.lower().strip()

def is_valid_input(text):
    return len(text.strip()) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

def fix_summary_capitalization(text):
    return ". ".join(sentence.strip().capitalize() for sentence in text.split(".") if sentence).strip() + "."

# --- Load IndoBERT ---
@st.cache_resource
def load_model():
    model_path = "models/indobert-fold0/checkpoint-3800"  # Update with your final/best model path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- UI Header ---
st.markdown("<h1 style='text-align: center;'>Deteksi Berita Hoaks 🔎</h1>", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("""
    <style>
    .custom-box {
        background-color: #ffffff;
        border: 2px solid #b52f2f;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .stTextArea textarea {
        background-color: #f9f9f9 !important;
        border: 1px solid #ccc !important;
        border-radius: 10px !important;
        padding: 15px !important;
        font-size: 16px !important;
    }
    .stButton > button {
        background-color: #b52f2f;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        margin-top: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ebebeb;
        border: 2px solid #b52f2f;
        transform: scale(1.04);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-box'><h3>📰 Masukkan Artikel Berita</h3>", unsafe_allow_html=True)

text = st.text_area(
    label="",
    height=200,
    placeholder="Petunjuk:\nMasukkan teks dari sumber online\nGunakan Bahasa Indonesia\nMinimal 30 karakter",
    label_visibility="collapsed"
)

submit = st.button("🔍 Periksa")
st.markdown("</div>", unsafe_allow_html=True)

# --- Predict with IndoBERT ---
if submit:
    text = text.strip()
    if not text:
        st.warning("⚠️ Tolong masukkan teks terlebih dahulu.")
    elif not is_valid_input(text):
        st.warning("⚠️ Teks terlalu pendek atau tidak valid. Masukkan minimal 30 karakter yang bermakna.")
    elif detect(text) != "id":
        st.warning("⚠️ Artikel harus menggunakan Bahasa Indonesia.")
    else:
        cleaned = clean_text(text)
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        st.markdown("## Hasil Deteksi")
        result_label = "🚨 **Hoax**" if pred == 1 else "✅ **Valid**"
        st.success(f"Hasil Deteksi: {result_label}")
        st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%}")
        
        if confidence < 0.60:
            st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")

        # Ringkasan
        st.markdown("### Ringkasan Artikel:")
        try:
            raw_summary = summarize(cleaned, words=80)
            summary_text = fix_summary_capitalization(raw_summary)
            if summary_text.strip():
                st.info(summary_text)
            else:
                st.info("Teks terlalu pendek untuk diringkas secara otomatis.")
        except Exception as e:
            st.warning(f"Gagal membuat ringkasan: {e}")

# --- Footer ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.markdown("""---""")

with st.expander("Tentang Aplikasi ℹ️", expanded=False):
    st.markdown("""
    Aplikasi ini digunakan untuk mendeteksi apakah sebuah artikel mengandung informasi hoaks atau tidak berdasarkan teks yang dimasukkan.
    """)

st.markdown(
    """
    <hr style='border-top: 1px solid #bbb;'>
    <div style='text-align: center; font-size: 14px;'>
        Dibuat oleh <b>Mohammad Ramadhan Zainoor</b> · © 2025 · <a href="https://github.com/zainoor" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True
)
