import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect
import re
from summa.summarizer import summarize

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")

# --- Helper ---
def clean_text(text):
    return text.lower().strip()

def is_valid_input(text):
    return len(text.strip()) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

# --- Load IndoBERT hoax-detector ---
@st.cache_resource
def load_classifier():
    model_path = "zainoor/indo-hoax-detector-indobert"
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_classifier()

def summarize_text(text):
    summary = summarize(text, ratio=0.2)
    return summary.strip() if summary else "Teks terlalu pendek untuk diringkas."

# ---------- UI ----------
st.markdown(
    "<h1 style='text-align: center;'>Deteksi Berita Hoaks 🔎 (IndoBERT)</h1>",
    unsafe_allow_html=True,
)

# CSS Styling
st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

# ----- Input -----
st.markdown("<div class='custom-box'><h3>📰 Masukkan Artikel Berita</h3>", unsafe_allow_html=True)

text = st.text_area(
    label="Masukkan teks artikel",
    height=200,
    placeholder="Petunjuk:\nMasukkan teks dari sumber online\nGunakan Bahasa Indonesia\nMinimal 30 karakter",
    label_visibility="collapsed",
)

submit = st.button("🔍 Periksa")
st.markdown("</div>", unsafe_allow_html=True)

# ----- Prediksi -----
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

            # ✅ Solusi: kalibrasi confidence menggunakan temperature scaling
            TEMPERATURE = 2.5  # Naikkan sesuai hasil eksperimen
            logits = outputs.logits

            # 🔧 Kurangi efek overconfidence dengan clipping + smoothing
            logits = torch.clamp(logits, -4, 4)  # Batasi range logit
            probs = torch.nn.functional.softmax(logits / 2.5, dim=1)  # scaling

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        def describe_confidence(score):
            if score > 0.95:
                return "Sangat Yakin 🔥"
            elif score > 0.85:
                return "Cukup Yakin 👍"
            elif score > 0.6:
                return "Yakin 🙂"
            else:
                return "Kurang Yakin 🤔"

        # ----- Hasil Deteksi -----
        st.markdown("## Hasil Deteksi")
        label = "🚨 **Hoax**" if pred == 1 else "✅ **Valid**"
        st.success(f"Hasil Deteksi: {label}")
        st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%} ({describe_confidence(confidence)})")

        st.progress(confidence)

        if confidence < 0.60:
            st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")

        # ----- Ringkasan -----
        st.markdown("### Ringkasan Artikel:")
        try:
            summary = summarize_text(cleaned)
            st.info(summary)
        except Exception as e:
            st.warning(f"Gagal membuat ringkasan: {e}")

# ----- Footer -----
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("Tentang Aplikasi ℹ️", expanded=False):
    st.markdown(
        """
        Aplikasi ini digunakan untuk mendeteksi apakah sebuah artikel mengandung informasi hoaks atau tidak berdasarkan teks yang dimasukkan.
        """
    )

st.markdown(
    """
    <hr style='border-top: 1px solid #bbb;'>
    <div style='text-align: center; font-size: 14px;'>
        Dibuat oleh <b>Mohammad Ramadhan Zainoor</b> · © 2025 ·
        <a href="https://github.com/zainoor" target="_blank">GitHub Repo</a>
    </div>
    """,
    unsafe_allow_html=True,
)
