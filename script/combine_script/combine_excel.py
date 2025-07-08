import pandas as pd
import os

# Load data
print("Loading data...")
politik = pd.read_csv("rawdata/politik.csv")
scraped = pd.read_csv("rawdata/scraped.csv")

print(f"Data awal - politik: {len(politik)}, scraped: {len(scraped)}")

# Cek kolom yang ada
print("\nKolom di politik.csv:", politik.columns.tolist())
print("Kolom di scraped.csv:", scraped.columns.tolist())

# Drop baris dengan data PENTING kosong (hanya Title, FullText, label)
# Author, Url, Date bisa kosong karena tidak krusial untuk training
critical_columns = ['Title', 'FullText', 'label']
print(f"\nSebelum dropna - politik: {len(politik)}, scraped: {len(scraped)}")

politik.dropna(subset=critical_columns, inplace=True)
scraped.dropna(subset=critical_columns, inplace=True)

print(f"Setelah dropna - politik: {len(politik)}, scraped: {len(scraped)}")

# Cek unique values di kolom label sebelum konversi
print("\nUnique values di label - politik:", politik['label'].unique())
print("Unique values di label - scraped:", scraped['label'].unique())

# Pastikan label dalam bentuk angka
try:
    politik['label'] = politik['label'].astype(int)
    scraped['label'] = scraped['label'].astype(int)
    print("✅ Konversi label berhasil")
except Exception as e:
    print(f"❌ Error konversi label: {e}")
    # Coba bersihkan label yang mungkin ada spasi atau karakter aneh
    politik['label'] = pd.to_numeric(politik['label'], errors='coerce')
    scraped['label'] = pd.to_numeric(scraped['label'], errors='coerce')
    
    # Drop baris dengan label yang tidak bisa dikonversi
    politik.dropna(subset=['label'], inplace=True)
    scraped.dropna(subset=['label'], inplace=True)
    
    politik['label'] = politik['label'].astype(int)
    scraped['label'] = scraped['label'].astype(int)

# Analisis distribusi per dataset
print("\n=== ANALISIS PER DATASET ===")
print("Politik.csv:")
print(f"  - Total: {len(politik)}")
print(f"  - Hoax (1): {len(politik[politik['label'] == 1])}")
print(f"  - Valid (0): {len(politik[politik['label'] == 0])}")

print("Scraped.csv:")
print(f"  - Total: {len(scraped)}")
print(f"  - Hoax (1): {len(scraped[scraped['label'] == 1])}")
print(f"  - Valid (0): {len(scraped[scraped['label'] == 0])}")

# Gabungkan semua data
all_data = pd.concat([politik, scraped], ignore_index=True)

# Pisahkan berdasarkan label
hoax_data = all_data[all_data['label'] == 1]  # hoax
valid_data = all_data[all_data['label'] == 0]  # valid

print(f"\n=== HASIL GABUNGAN ===")
print(f"Total data hoax: {len(hoax_data)}")
print(f"Total data valid: {len(valid_data)}")

# Tentukan jumlah target
target_count = 8000
available_min = min(len(hoax_data), len(valid_data))
min_count = min(available_min, target_count)

print(f"\nTarget per kelas: {target_count}")
print(f"Jumlah yang akan digunakan: {min_count}")

if min_count < target_count:
    print(f"⚠️  PERINGATAN: Data tidak mencukupi untuk target {target_count}")

# Sampling data seimbang
sample_hoax = hoax_data.sample(n=min_count, random_state=42)
sample_valid = valid_data.sample(n=min_count, random_state=42)

# Gabungkan dataset seimbang
final_df = pd.concat([sample_hoax, sample_valid], ignore_index=True)

# Shuffle dataset final
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan ke CSV
os.makedirs("rawdata", exist_ok=True)
final_df.to_csv("rawdata/combine1.csv", index=False)

print("\n✅ Data seimbang berhasil disimpan.")
print("\nDistribusi label final:")
print(final_df['label'].value_counts())
print(f"Total data: {len(final_df)}")

# Verifikasi distribusi berdasarkan source
print("\nDistribusi berdasarkan source:")
print(final_df.groupby(['source', 'label']).size().unstack(fill_value=0))