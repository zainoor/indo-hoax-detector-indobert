import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\b(?:com|www|http|https|twitter|facebook|tiktok|instagram|youtube)\b", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load dataset
def load_dataset(path_csv):
    df = pd.read_csv(path_csv)
    df["cleaned"] = df["cleaned"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df[["cleaned", "label"]])

# Tokenization
def tokenize_dataset(dataset, tokenizer_name="indobenchmark/indobert-base-p1", max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    def tokenize(batch):
        return tokenizer(batch["cleaned"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    return tokenized.remove_columns(["cleaned"]).with_format("torch")

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Logging callback
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:  # Simpan semua log
            self.logs.append(logs)

    def to_dataframe(self):
        return pd.DataFrame(self.logs)

# Training
def train_model(dataset, model_name="indobenchmark/indobert-base-p1", output_dir="trained_model"):
    # ❗ Bersihkan folder output jika sudah ada
    if os.path.exists(output_dir):
        print(f"🧹 Cleaning existing folder: {output_dir}")
        shutil.rmtree(output_dir)

    df = dataset.to_pandas()
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["labels"], random_state=42)
    train_ds = Dataset.from_pandas(train_df).remove_columns(["__index_level_0__"])
    val_ds = Dataset.from_pandas(val_df).remove_columns(["__index_level_0__"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",        
        eval_steps=100,               # evaluasi tiap 100 langkah
        save_strategy="epoch",
        save_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0
    )

    callback = LoggingCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), callback]
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("📊 Final Evaluation:", metrics)

    # Simpan model terbaik
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    trainer.model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"✅ Best model saved to: {best_model_dir}")

    # Simpan log ke CSV
    log_df = callback.to_dataframe()
    log_df.to_csv(os.path.join(output_dir, "training_log.csv"), index=False)

    # Grafik Loss
    if 'loss' in log_df.columns:
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=log_df["epoch"], y=log_df["loss"], label="Train Loss")
        if 'eval_loss' in log_df.columns:
            sns.lineplot(x=log_df["epoch"], y=log_df["eval_loss"], label="Val Loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
        plt.close()

    # Grafik F1 Score
    if 'eval_f1' in log_df.columns:
        plt.figure(figsize=(6, 4))
        sns.lineplot(x=log_df["epoch"], y=log_df["eval_f1"], label="Val F1")
        plt.title("F1 Score per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_curve.png"))
        plt.close()

    return trainer

# 7. Main
def main():
    print("🔎 Loading and preprocessing data...")
    dataset = load_dataset("cleandata/train_dataset.csv")
    print("🧠 Tokenizing...")
    dataset = tokenize_dataset(dataset)
    print("🚀 Starting training...")
    trainer = train_model(dataset)
    print("✅ Done.")

if __name__ == "__main__":
    main()
