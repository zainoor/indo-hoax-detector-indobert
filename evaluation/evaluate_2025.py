# evaluate_2025.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch.utils.data import Dataset

# Load model dan tokenizer dari model terbaik
model_path = "models/indobert-fold0/checkpoint-3800"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
trainer = Trainer(model=model)

# Load data test 2025
df = pd.read_csv("cleandata/hoax_dataset_2025.csv")
df = df.dropna(subset=["cleaned", "label"])

texts = df["cleaned"].tolist()
labels = df["label"].tolist()

# Tokenisasi
encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

# Tambahkan di atas sebelum Trainer.predict()
class HoaxDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Buat dataset untuk evaluasi
dataset = HoaxDataset(encodings, labels)

# Prediksi
outputs = trainer.predict(dataset)
preds = np.argmax(outputs.predictions, axis=1)

# Confusion matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Valid", "Hoax"], yticklabels=["Valid", "Hoax"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix on 2025 Data")
plt.savefig("results/confusion_matrix_2025_new.png")
plt.show()

# Classification report
report = classification_report(labels, preds, target_names=["Valid", "Hoax"])
print(report)
with open("results/classification_report_2025_new.txt", "w") as f:
    f.write(report)
