import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Ensure base directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Load and balance dataset
df = pd.read_csv("cleandata/hoax_dataset_merged.csv")
df = df.dropna(subset=["cleaned", "label"])

# Undersample to balance
hoax_df = df[df["label"] == 1]
valid_df = df[df["label"] == 0].sample(n=len(hoax_df), random_state=42)
df = pd.concat([hoax_df, valid_df]).sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["cleaned"].tolist()
labels = df["label"].tolist()

# Tokenizer
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization helper
def tokenize_with_progress(texts, tokenizer, max_length=256):
    encodings = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
    for text in tqdm(texts, desc="Tokenizing texts"):
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        encodings["input_ids"].append(enc["input_ids"].squeeze().tolist())
        encodings["attention_mask"].append(enc["attention_mask"].squeeze().tolist())
        encodings["token_type_ids"].append(enc["token_type_ids"].squeeze().tolist())
    return encodings

# Custom Dataset
def make_dataset(encodings, labels):
    return Dataset.from_dict({**encodings, "label": labels})

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Save confusion matrix plot
def save_confusion_matrix(y_true, y_pred, fold, labels=["Valid", "Hoax"]):
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix_fold{fold}.png")
    plt.close()

# Save classification report
def save_classification_report(y_true, y_pred, fold):
    os.makedirs("results", exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=["Valid", "Hoax"])
    with open(f"results/classification_report_fold{fold}.txt", "w") as f:
        f.write(report)

# K-Fold setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_index, val_index) in enumerate(skf.split(texts, labels)):
    print(f"\n==== Fold {fold + 1} ====")

    # Split data
    X_train, X_val = [texts[i] for i in train_index], [texts[i] for i in val_index]
    y_train, y_val = [labels[i] for i in train_index], [labels[i] for i in val_index]

    # Tokenize
    train_encodings = tokenize_with_progress(X_train, tokenizer)
    val_encodings = tokenize_with_progress(X_val, tokenizer)

    train_dataset = make_dataset(train_encodings, y_train)
    val_dataset = make_dataset(val_encodings, y_val)

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Ensure model/log dirs
    model_output_dir = f"./models/indobert-fold{fold}"
    log_dir = f"./logs/fold{fold}"
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=log_dir,
        logging_steps=10,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train() #resume_from_checkpoint=True

    # Evaluate
    train_preds = trainer.predict(train_dataset)
    val_preds = trainer.predict(val_dataset)

    y_train_pred = np.argmax(train_preds.predictions, axis=1)
    y_val_pred = np.argmax(val_preds.predictions, axis=1)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = precision_recall_fscore_support(y_val, y_val_pred, average='binary')[2]

    print(f"Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f}")
    print("Validation Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Valid", "Hoax"]))

    save_confusion_matrix(y_val, y_val_pred, fold + 1)
    save_classification_report(y_val, y_val_pred, fold + 1)

    fold_metrics.append({
        "fold": fold + 1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_f1": val_f1
    })

# Save fold metrics
os.makedirs("results", exist_ok=True)
with open("results/kfold_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fold", "train_accuracy", "val_accuracy", "val_f1"])
    writer.writeheader()
    writer.writerows(fold_metrics)

# Summary
print("\n==== Cross-Validation Summary ====")
for m in fold_metrics:
    print(f"Fold {m['fold']}: Train Acc = {m['train_accuracy']:.4f} | Val Acc = {m['val_accuracy']:.4f} | Val F1 = {m['val_f1']:.4f}")
