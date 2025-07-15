import pandas as pd
import os

VALID_PATH = "rawdata/valid_news.csv"
HOAX_PATH = "rawdata/hoax_news.csv"
OUTPUT_PATH = "rawdata/combine_news_balanced.csv"

os.makedirs("rawdata", exist_ok=True)

try:
    df_valid = pd.read_csv(VALID_PATH)
    df_hoax = pd.read_csv(HOAX_PATH)

    # Undersample both to match the smaller class
    n_samples = min(len(df_valid), len(df_hoax))
    df_valid_balanced = df_valid.sample(n=n_samples, random_state=42)
    df_hoax_balanced = df_hoax.sample(n=n_samples, random_state=42)

    # Combine and shuffle
    df_all = pd.concat([df_valid_balanced, df_hoax_balanced], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Sanitize weird Unicode line breaks
    df_all["Text"] = df_all["Text"].astype(str).str.replace('\u2028', ' ', regex=False)
    df_all["Text"] = df_all["Text"].str.replace('\u2029', ' ', regex=False)
    df_all["Text"] = df_all["Text"].str.replace('\r', ' ', regex=False)
    df_all["Text"] = df_all["Text"].str.replace('\n', ' ', regex=False)

    # Save
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Combined (undersampled) dataset saved → {OUTPUT_PATH}")
    print(f"📊 Final size: {len(df_all)} rows ({n_samples} valid + {n_samples} hoax)")

except Exception as e:
    print(f"❌ Error combining datasets: {e}")
