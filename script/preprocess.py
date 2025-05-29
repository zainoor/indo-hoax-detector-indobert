import pandas as pd
import re
import os
from tqdm import tqdm
from unidecode import unidecode

tqdm.pandas()

# Cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = unidecode(text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Folder path
data_path = "rawdata"

# Read XLSX files with tqdm
def process_xlsx(file, text_col, label_col):
    print(f"📥 Loading {file}.xlsx...")
    df = pd.read_excel(os.path.join(data_path, file + ".xlsx"))
    df = df[[text_col, label_col]]
    df.columns = ["text", "label"]
    return df

print("🔄 Reading all files...")

cnn = process_xlsx("cnn", "text_new", "hoax")
kompas = process_xlsx("kompas", "text_new", "hoax")
tempo = process_xlsx("tempo", "text_new", "hoax")
turnbackhoax = process_xlsx("turnbackhoax", "Narasi", "hoax")

hoax_valid = pd.read_csv(os.path.join(data_path, "hoax_valid_labeled.csv"))
hoax_valid = hoax_valid[["berita", "label"]].rename(columns={"berita": "text"})

politik = pd.read_csv(os.path.join(data_path, "politik.csv"))
politik["text"] = politik["judul"].fillna('') + " " + politik["narasi"].fillna('')
politik = politik[["text", "label"]]

# Combine
all_data = pd.concat([cnn, kompas, tempo, turnbackhoax, hoax_valid, politik], ignore_index=True)

# Drop missing
all_data = all_data.dropna(subset=["text", "label"])

# Clean text with tqdm
print("🧹 Cleaning text...")
all_data["cleaned"] = all_data["text"].progress_apply(clean_text)

# Final output
final_df = all_data[["cleaned", "label"]].dropna().reset_index(drop=True)

# Save
final_df.to_csv("cleandata/hoax_dataset_merged.csv", index=False)
print("✅ Done. Final shape:", final_df.shape)
