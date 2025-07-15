import pandas as pd
import re
import os

INPUT = "rawdata/combine_news_balanced.csv"
FINAL_FULL = "rawdata/combine_news_final.csv"
TRAIN_ONLY = "cleandata/train_dataset.csv"

os.makedirs("rawdata", exist_ok=True)
os.makedirs("cleandata", exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    # --- HOAX TAGS ---
    hoax_patterns = [
        r"\[(SALAH|DISINFORMASI|NARASI|BENAR|PENIPUAN|HOAKS|HOAX)\]:?",
        r"\b(KLARIFIKASI|HOAKS|HOAX|DISINFORMASI|HASUT|HOAX\s*\+\s*HASUT|ISU|FITNAH|PENIPUAN|EDUKASI)\b",
        r"\((DISINFORMASI|MISINFORMASI|EDUKASI)\)"
    ]
    for pat in hoax_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # --- VALID LAYOUT ARTIFACTS ---
    valid_noise = [
        r"ADVERTISEMENT.*?CONTINUE WITH CONTENT",
        r"SCROLL TO RESUME CONTENT",
        r"Lihat Juga\s*:", r"Baca Juga\s*:",
        r"\[Gambas:.*?\]", r"\[VIDEO\]", r"\[FULL\]",
        r"\[POPULER NASIONAL\]", r"\[POPULER NUSANTARA\]", r"\[HOAKS\]",
        r"CNN Indonesia", r"TEMPO\.CO", r"KOMPAS\.com", r"\(fnr/bmw\)"
    ]
    for pat in valid_noise:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # --- Normalize punctuation and spacing ---
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\r|\n|\u2028|\u2029", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# --- Load dataset ---
df = pd.read_csv(INPUT)
print(f"✅ Loaded {len(df)} rows from {INPUT}")

# --- Clean and filter ---
df["Text"] = df["Text"].astype(str).apply(clean_text)
df = df[df["Text"].str.len() >= 20].dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

# --- Save outputs ---
df.to_csv(FINAL_FULL, index=False)
print(f"✅ Cleaned full dataset saved → {FINAL_FULL} ({len(df)} rows)")

df[["Text", "label"]].to_csv(TRAIN_ONLY, index=False)
print(f"✅ Training dataset saved → {TRAIN_ONLY} ({len(df)} rows)")
