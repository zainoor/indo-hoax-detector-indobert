import pandas as pd
import re
import html
from transformers import AutoTokenizer

# Load IndoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

# Patterns to clean
NOISY_TAGS = [
    r"\[SALAH\]", r"\[DISINFORMASI\]", r"\[HOAKS\]", r"\[NARASI[^\]]*\]",
    r"\[Fakta\]", r"\[BENAR\]", r"\[KLARIFIKASI\]", r"\[FITNAH\]",
    r"\[PENIPUAN\]", r"\[ISU\]", r"\[MISINFORMASI\]", r"\(EDUKASI\)",
    r"HOAX", r"HASUT", r"HOAX \+ HASUT", r"\(DISINFORMASI/MISINFORMASI\)"
]

LAYOUT_ARTIFACTS = [
    r"ADVERTISEMENT\s+SCROLL\s+TO\s+CONTINUE",
    r"ADVERTISEMENT", r"IKLAN", r"SCROLL TO CONTINUE",
    r"ikuti berita terkini dari.*?klik di sini",  # Tempo footer
]

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities and remove broken emoji encodings
    text = html.unescape(text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove noisy tags
    for tag in NOISY_TAGS:
        text = re.sub(tag, '', text, flags=re.IGNORECASE)

    # Remove layout/boilerplate artifacts
    for artifact in LAYOUT_ARTIFACTS:
        text = re.sub(artifact, '', text, flags=re.IGNORECASE)

    # Remove twitter mentions and URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Normalize quotes and spaces
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()

def truncate_text(text, max_tokens=480):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

def preprocess_dataframe(df, text_col="Text"):
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[text_col] = df[text_col].apply(truncate_text)
    return df

def main():
    # Load dataset
    input_path = "cleandata/train_dataset.csv"
    output_path = "cleandata/train_dataset_cleaned.csv"

    print("🔄 Loading dataset...")
    df = pd.read_csv(input_path)

    # Confirm required column exists
    if "Text" not in df.columns:
        raise ValueError("Dataset must contain a 'Text' column.")

    print("✨ Preprocessing...")
    df_cleaned = preprocess_dataframe(df, text_col="Text")

    print("💾 Saving cleaned dataset...")
    df_cleaned.to_csv(output_path, index=False)
    print(f"✅ Done! Saved to {output_path}")

if __name__ == "__main__":
    main()
