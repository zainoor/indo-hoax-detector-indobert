import os
import pandas as pd

HOAX_DIR = "rawdata/hoax"
OUTPUT_PATH = "rawdata/hoax_news.csv"
os.makedirs("rawdata", exist_ok=True)

def clean_text(text):
    if isinstance(text, str):
        return text.replace('\u2028', ' ').replace('\u2029', ' ').replace('\r', ' ').replace('\n', ' ').strip()
    return ""

def normalize_hoax(df):
    text_col = df["Narasi"] if "Narasi" in df.columns else df["cleaned"]
    return pd.DataFrame({
        "Title": df["Title"] if "Title" in df.columns else ["" for _ in range(len(df))],
        "Text": text_col,
        "Author": df["Author"] if "Author" in df.columns else ["" for _ in range(len(df))],
        "Url": df["Url"] if "Url" in df.columns else ["" for _ in range(len(df))],
        "Date": df["Date"] if "Date" in df.columns else ["" for _ in range(len(df))],
        "label": 1,
        "source": "hoax"
    })

hoax_dfs = []
for fname in os.listdir(HOAX_DIR):
    fpath = os.path.join(HOAX_DIR, fname)
    try:
        df = pd.read_csv(fpath) if fname.endswith(".csv") else pd.read_excel(fpath)
        col = "Narasi" if "Narasi" in df.columns else "cleaned"
        df = df.dropna(subset=[col])
        print(f"✅ Loaded {len(df)} rows from {fname}")
        df["cleaned"] = df[col]
        hoax_dfs.append(normalize_hoax(df))
    except Exception as e:
        print(f"❌ {fname}: {e}")

if hoax_dfs:
    print("🔍 Final row count before dropping duplicates:", len(pd.concat(hoax_dfs)))
    df_hoax = pd.concat(hoax_dfs, ignore_index=True).drop_duplicates(subset=["Text"])
    df_hoax.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df_hoax)} hoax entries → {OUTPUT_PATH}")
else:
    print("⚠️ No hoax files processed.")
