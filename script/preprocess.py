import pandas as pd

# 1. Baca file
politik_df = pd.read_csv("rawdata/politik.csv")
scraped_df = pd.read_csv("rawdata/scraped.csv")

# 2. Bersihkan baris kosong atau yang tidak lengkap
politik_df.dropna(subset=['Title', 'FullText', 'Author', 'Url', 'Date', 'label', 'source'], inplace=True)
scraped_df.dropna(subset=['Title', 'FullText', 'Author', 'Url', 'Date', 'label', 'source'], inplace=True)

# 3. Seimbangkan data dari politik.csv berdasarkan kolom label (misal: "hoax" dan "valid")
min_count = politik_df['label'].value_counts().min()
balanced_politik_df = (
    politik_df.groupby('label')
    .sample(n=min_count, random_state=42)
    .reset_index(drop=True)
)

# 4. Gabungkan data balanced politik dan scraped
combined_df = pd.concat([balanced_politik_df, scraped_df], ignore_index=True)

# 5. Simpan hasilnya
combined_df.to_csv("rawdata/hasil_gabungan_seimbang.csv", index=False)

print(f"Gabungan selesai. Jumlah total data: {len(combined_df)}")
