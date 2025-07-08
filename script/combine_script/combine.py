import pandas as pd
import os

# Load data
politik = pd.read_csv("rawdata/politik.csv")
scraped = pd.read_csv("rawdata/scraped.csv")

# Drop baris dengan data penting kosong
required_columns = ['Title', 'FullText', 'Author', 'Url', 'Date', 'label', 'source']
politik.dropna(subset=required_columns, inplace=True)
scraped.dropna(subset=required_columns, inplace=True)

# Pastikan label dalam bentuk angka
politik['label'] = politik['label'].astype(int)
scraped['label'] = scraped['label'].astype(int)

# Gabungkan semua data dulu
all_data = pd.concat([politik, scraped], ignore_index=True)

# Pisahkan berdasarkan label
hoax_data = all_data[all_data['label'] == 1]  # hoax
valid_data = all_data[all_data['label'] == 0]  # valid

print(f"Total data hoax: {len(hoax_data)}")
print(f"Total data valid: {len(valid_data)}")

# Tentukan jumlah target untuk balancing
target_count = 8000

print(f"Data hoax tersedia: {len(hoax_data)}")
print(f"Data valid tersedia: {len(valid_data)}")
print(f"Target per kelas: {target_count}")

# Periksa apakah data mencukupi
if len(hoax_data) < target_count:
    print(f"⚠️  PERINGATAN: Data hoax hanya {len(hoax_data)}, kurang dari target {target_count}")
    min_count = len(hoax_data)
elif len(valid_data) < target_count:
    print(f"⚠️  PERINGATAN: Data valid hanya {len(valid_data)}, kurang dari target {target_count}")
    min_count = len(valid_data)
else:
    print(f"✅ Data mencukupi untuk target {target_count}")
    min_count = target_count

print(f"Jumlah yang akan digunakan untuk setiap kelas: {min_count}")

# Sampling data seimbang
sample_hoax = hoax_data.sample(n=min_count, random_state=42)
sample_valid = valid_data.sample(n=min_count, random_state=42)

# Gabungkan dataset seimbang
final_df = pd.concat([sample_hoax, sample_valid], ignore_index=True)

# Shuffle dataset final
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan ke CSV
os.makedirs("rawdata", exist_ok=True)
final_df.to_csv("rawdata/combine.csv", index=False)

print("✅ Data seimbang berhasil disimpan.")
print("\nDistribusi label final:")
print(final_df['label'].value_counts())
print(f"\nTotal data: {len(final_df)}")

# Verifikasi distribusi berdasarkan source
print("\nDistribusi berdasarkan source:")
print(final_df.groupby(['source', 'label']).size().unstack(fill_value=0))