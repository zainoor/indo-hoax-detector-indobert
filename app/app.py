import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification,BertForSequenceClassification
import torch
from langdetect import detect
import re

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="Deteksi Berita Hoaks", layout="centered")

# --- Helper ---
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9À-ÿ\s.,!?;:'\"()-]", "", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_input(text):
    return len(text.strip()) >= 30 and re.search(r"[a-zA-Z]{3,}", text)

# --- Load IndoBERT hoax-detector ---
@st.cache_resource
def load_classifier():
    try:
        model_id = "zainoor/indo-hoax-detector-indobert-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Simple text summarization function (alternative to summa)
def summarize_text(text, max_sentences=3):
    """
    Simple extractive summarization by selecting first few sentences
    """
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Return first few sentences as summary
        summary = '. '.join(sentences[:max_sentences]) + '.'
        return summary
    except Exception as e:
        return "Gagal membuat ringkasan."

# Advanced summarization using summa (fallback)
def summarize_with_summa(text):
    """
    Try to use summa library for summarization
    """
    try:
        from summa.summarizer import summarize
        summary = summarize(text, ratio=0.2)
        return summary.strip() if summary else "Teks terlalu pendek untuk diringkas."
    except ImportError:
        st.warning("Library summa tidak tersedia. Menggunakan metode ringkasan sederhana.")
        return summarize_text(text)
    except Exception as e:
        st.warning(f"Error dengan summa: {e}. Menggunakan metode ringkasan sederhana.")
        return summarize_text(text)

# Load model
tokenizer, model = load_classifier()

# Check if model loaded successfully
if tokenizer is None or model is None:
    st.error("Gagal memuat model. Pastikan koneksi internet stabil dan coba lagi.")
    st.stop()

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
    else:
        # Check language
        try:
            if detect(text) != "id":
                st.warning("⚠️ Artikel harus menggunakan Bahasa Indonesia.")
            else:
                cleaned = clean_text(text)
                inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)

                with torch.no_grad():
                    outputs = model(**inputs)
                    TEMPERATURE = 1.5
                    logits = outputs.logits / TEMPERATURE
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred].item()

                def describe_confidence(score):
                    if score > 0.95:
                        return "Sangat Yakin"
                    elif score > 0.85:
                        return "Cukup Yakin"
                    elif score > 0.6:
                        return "Yakin"
                    else:
                        return "Kurang Yakin"

                st.markdown("## Hasil Deteksi")
                label = "🚨 **Hoax**" if pred == 1 else "✅ **Valid**"
                st.success(f"Hasil Deteksi: {label}")
                st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%} ({describe_confidence(confidence)})")

                if confidence < 0.60:
                    st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")

                # ----- Ringkasan -----
                st.markdown("### Ringkasan Artikel:")
                try:
                    summary = summarize_with_summa(cleaned)
                    st.info(summary)
                except Exception as e:
                    st.warning(f"Gagal membuat ringkasan: {e}")
        
        except Exception as e:
            st.error(f"Error dalam deteksi bahasa: {e}")
            st.info("Melanjutkan tanpa pengecekan bahasa...")
            
            # Continue with prediction without language check
            cleaned = clean_text(text)
            inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)

            with torch.no_grad():
                outputs = model(**inputs)
                TEMPERATURE = 1.5
                logits = outputs.logits / TEMPERATURE
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

            def describe_confidence(score):
                if score > 0.95:
                    return "Sangat Yakin"
                elif score > 0.85:
                    return "Cukup Yakin"
                elif score > 0.6:
                    return "Yakin"
                else:
                    return "Kurang Yakin"

            st.markdown("## Hasil Deteksi")
            label = "🚨 **Hoax**" if pred == 1 else "✅ **Valid**"
            st.success(f"Hasil Deteksi: {label}")
            st.markdown(f"**Tingkat Keyakinan:** {confidence:.2%} ({describe_confidence(confidence)})")

            if confidence < 0.60:
                st.warning("⚠️ Hasil deteksi kurang meyakinkan. Harap verifikasi ulang informasi ini.")

            # ----- Ringkasan -----
            st.markdown("### Ringkasan Artikel:")
            try:
                summary = summarize_with_summa(cleaned)
                st.info(summary)
            except Exception as e:
                st.warning(f"Gagal membuat ringkasan: {e}")

# ----- Footer -----
# st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
# st.markdown("---")

# with st.expander("Tentang Aplikasi ℹ️", expanded=False):
#     st.markdown(
#         """
#         Aplikasi ini digunakan untuk mendeteksi apakah sebuah artikel mengandung informasi hoaks atau tidak berdasarkan teks yang dimasukkan.
#         """
#     )

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