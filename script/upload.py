from huggingface_hub import create_repo, upload_folder

# Nama akun dan repo
repo_id = "zainoor/indo-hoax-detector-indobert-v3"

# Path ke model
local_model_dir = "saved_model/fold_5"

# 1. Buat repo (kalau belum ada)
create_repo(repo_id=repo_id, exist_ok=True)

# 2. Upload isi folder model
upload_folder(
    repo_id=repo_id,
    folder_path=local_model_dir,
    path_in_repo=".",  # Upload ke root repo
    commit_message="Upload IndoBERT Hoax Detector v3"
)

print(f"✅ Selesai! Model kamu bisa diakses di: https://huggingface.co/{repo_id}")
