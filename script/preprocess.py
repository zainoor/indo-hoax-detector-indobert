import pandas as pd
import re

# BACA FILE CSV (ganti dengan nama file kamu)
df = pd.read_csv("rawdata/combine.csv")  # <- ganti sesuai nama file asli

# FUNGSI CLEANING
def clean_text(text):
    text = str(text).lower()  # case folding
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # hapus URL
    text = re.sub(r'\b(?:com|www|http|https|twitter|facebook|tiktok|instagram|youtube)\b', '', text)  # hapus kata platform
    text = re.sub(r'@\w+', '', text)  # hapus mention
    text = re.sub(r'#\w+', '', text)  # hapus hashtag
    text = re.sub(r'\([^)]*\)', '', text)  # hapus isi dalam tanda kurung
    text = re.sub(r'\s+', ' ', text).strip()  # hapus spasi berlebih
    return text

# CLEANING
df = df[['FullText', 'label']].dropna()  # pastikan kolom tersedia
df['cleaned'] = df['FullText'].apply(clean_text)

# PILIH HANYA 2 KOLOM SAJA
df_final = df[['cleaned', 'label']]

# SIMPAN KE FILE BARU
df_final.to_csv("cleandata/train_dataset.csv", index=False)

print("✅ File berhasil dibuat: cleandata/train_dataset.csv")
print(df_final.head())
