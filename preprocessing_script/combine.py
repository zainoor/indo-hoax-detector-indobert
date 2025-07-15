import pandas as pd
import os

VALID_PATH = "rawdata/valid_news.csv"
HOAX_PATH = "rawdata/hoax_news.csv"
OUTPUT_PATH = "rawdata/combine_news.csv"

# Make sure output directory exists
os.makedirs("rawdata", exist_ok=True)

try:
    # Load valid and hoax datasets
    df_valid = pd.read_csv(VALID_PATH)
    df_hoax = pd.read_csv(HOAX_PATH)

    # Combine
    df_all = pd.concat([df_valid, df_hoax], ignore_index=True)

    # Shuffle
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Combined dataset saved → {OUTPUT_PATH} ({len(df_all)} rows)")

except Exception as e:
    print(f"❌ Error combining datasets: {e}")
