import pandas as pd
import os

TRUNCATED_PATH = "rawdata/combine_news_final_truncated.csv"
OUTPUT_PATH = "cleandata/train_dataset.csv"
os.makedirs("cleandata", exist_ok=True)

# Load truncated dataset
df = pd.read_csv(TRUNCATED_PATH)

# Keep only necessary columns
df_train = df[["Text", "label"]].dropna()
df_train["label"] = df_train["label"].astype(int)

# Save to training path
df_train.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Updated training dataset saved → {OUTPUT_PATH}")
print(f"📊 Total rows: {len(df_train)}")
