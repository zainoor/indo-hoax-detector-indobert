import os
import pandas as pd

VALID_DIR = "rawdata/valid"
OUTPUT_PATH = "rawdata/valid_news.csv"
os.makedirs("rawdata", exist_ok=True)

def normalize_valid(df):
    return pd.DataFrame({
        "Title": df["Title"] if "Title" in df.columns else ["" for _ in range(len(df))],
        "Text": df["cleaned"] if "cleaned" in df.columns else df["FullText"],
        "Author": df["Author"] if "Author" in df.columns else ["" for _ in range(len(df))],
        "Url": df["Url"] if "Url" in df.columns else ["" for _ in range(len(df))],
        "Date": df["Date"] if "Date" in df.columns else ["" for _ in range(len(df))],
        "label": 0,
        "source": "valid"
    })

valid_dfs = []
for fname in os.listdir(VALID_DIR):
    fpath = os.path.join(VALID_DIR, fname)
    try:
        df = pd.read_csv(fpath) if fname.endswith(".csv") else pd.read_excel(fpath)
        col = "cleaned" if "cleaned" in df.columns else "FullText"
        df = df.dropna(subset=[col])
        df["cleaned"] = df[col]
        valid_dfs.append(normalize_valid(df))
    except Exception as e:
        print(f"❌ {fname}: {e}")

if valid_dfs:
    df_valid = pd.concat(valid_dfs, ignore_index=True).drop_duplicates(subset=["Text"])
    df_valid.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df_valid)} valid entries → {OUTPUT_PATH}")
else:
    print("⚠️ No valid files processed.")
