import pandas as pd
import re

# Kolom standar
standard_columns = ['Title', 'FullText', 'Author', 'Url', 'Date']

# Fungsi normalisasi kolom dan struktur
def normalize_df(df: pd.DataFrame, label: str = None):
    df.columns = [c.strip().lower() for c in df.columns]

    column_map = {}
    for col in df.columns:
        if 'judul' in col or 'title' in col:
            column_map[col] = 'Title'
        elif 'isi' in col or 'konten' in col or 'full' in col or 'teks' in col:
            column_map[col] = 'FullText'
        elif 'penulis' in col or 'author' in col:
            column_map[col] = 'Author'
        elif 'url' in col or 'tautan' in col or 'link' in col:
            column_map[col] = 'Url'
        elif 'tanggal' in col or 'date' in col or 'waktu' in col:
            column_map[col] = 'Date'
        elif 'label' in col:
            column_map[col] = 'label'

    df = df.rename(columns=column_map)

    # Pastikan semua kolom standar ada
    for col in standard_columns:
        if col not in df.columns:
            df[col] = ''

    # Tambahkan label jika belum ada
    if 'label' not in df.columns and label is not None:
        df['label'] = label

    return df[standard_columns + ['label']]

# Fungsi bersihkan karakter aneh dari teks
def clean_unusual_chars(text):
    # Hilangkan karakter line separator, paragraph separator, dan lainnya
    return re.sub(r'[\u2028\u2029\u000b\u000c]', '', str(text))

# Baca dan normalisasi data
kompas_df = normalize_df(pd.read_excel("rawdata/kompas.xlsx"), "valid")
tempo_df = normalize_df(pd.read_excel("rawdata/tempo.xlsx"), "valid")
cnn_df = normalize_df(pd.read_excel("rawdata/cnn.xlsx"), "valid")
turnbackhoax_df = normalize_df(pd.read_excel("rawdata/turnbackhoax.xlsx"), "hoax")

raw_hoax = normalize_df(pd.read_csv("rawdata/raw_hoax_2025_scrap.csv"))
raw_valid = normalize_df(pd.read_csv("rawdata/raw_valid_2025_scrap.csv"))

# Gabungkan semua
final_df = pd.concat([
    kompas_df,
    tempo_df,
    cnn_df,
    turnbackhoax_df,
    raw_hoax,
    raw_valid
], ignore_index=True)

# Konversi label ke angka
final_df['label'] = final_df['label'].map({'hoax': 1, 'valid': 0})

# Hapus karakter aneh dari kolom teks
for col in ['Title', 'FullText', 'Author']:
    final_df[col] = final_df[col].apply(clean_unusual_chars)

# Simpan ke file
final_df.to_csv("rawdata/dataset_combine.csv", index=False, lineterminator='\n')

print("✅ Dataset final berhasil disimpan sebagai 'hoax_dataset_2025_final.csv' tanpa karakter aneh dan duplikat.")
