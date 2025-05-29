import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tqdm import tqdm

# Load and balance dataset
df = pd.read_csv("cleandata/hoax_dataset_merged.csv")
df = df.dropna(subset=["cleaned", "label"])

# Undersample to balance
hoax_df = df[df["label"] == 1]
valid_df = df[df["label"] == 0].sample(n=len(hoax_df), random_state=42)
df = pd.concat([hoax_df, valid_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["cleaned"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# Load IndoBERT tokenizer
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize with tqdm progress bar
def tokenize_with_progress(texts, tokenizer, max_length=256):
    encodings = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
    for text in tqdm(texts, desc="Tokenizing texts"):
        enc = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        encodings["input_ids"].append(enc["input_ids"].squeeze().tolist())
        encodings["attention_mask"].append(enc["attention_mask"].squeeze().tolist())
        encodings["token_type_ids"].append(enc["token_type_ids"].squeeze().tolist())
    return encodings

train_encodings = tokenize_with_progress(train_texts, tokenizer)
test_encodings = tokenize_with_progress(test_texts, tokenizer)

# Convert to Dataset format
train_dataset = Dataset.from_dict({**train_encodings, "label": train_labels})
test_dataset = Dataset.from_dict({**test_encodings, "label": test_labels})

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training args with corrected parameters
training_args = TrainingArguments(
    output_dir="./models/indobert-hoax",
    eval_strategy="epoch",  # Correct parameter
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Define metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train with progress bar (handled by Trainer)
trainer.train()

# Save model and tokenizer
trainer.save_model("./models/indobert-hoax")
tokenizer.save_pretrained("./models/indobert-hoax")