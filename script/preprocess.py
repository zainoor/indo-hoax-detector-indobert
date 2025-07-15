import pandas as pd
import re

df = pd.read_csv("rawdata/combine.csv")

# Fungsi Cleaning
def clean_text(text):
    text = str(text).lower()  # case folding
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  
    text = re.sub(r'\b(?:com|www|http|https|twitter|facebook|tiktok|instagram|youtube)\b', '', text)  
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'\([^)]*\)', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

# Cleaning
df = df[['FullText', 'label']].dropna()  
df['cleaned'] = df['FullText'].apply(clean_text)

# Memilih 2 Kolom Yang Penting
df_final = df[['cleaned', 'label']]

# Simpan Ke File Baru
df_final.to_csv("cleandata/train_dataset.csv", index=False)

print("File berhasil dibuat: cleandata/train_dataset.csv")
print(df_final.head())
