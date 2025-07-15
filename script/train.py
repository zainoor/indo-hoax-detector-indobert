import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          Trainer, TrainingArguments, EarlyStoppingCallback,
                          DataCollatorWithPadding)
from datasets import Dataset
import torch
from scipy.special import softmax

# --- Config ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATA_PATH = "cleandata/train_dataset.csv"
RESULT_DIR = "new_results"
NUM_FOLDS = 5
BATCH_SIZE = 8 
EPOCHS = 10
PATIENCE = 2
SEED = 42

os.makedirs(RESULT_DIR, exist_ok=True)

# --- Load & Prepare Dataset ---
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["cleaned", "label"])
df["label"] = df["label"].astype(int)

# Check for duplicates
dupes = df.duplicated(subset=["cleaned"])
print("🔍 Duplicates in dataset:", dupes.sum())

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["cleaned"], truncation=True, padding=True, max_length=512)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥️ Current device:", torch.cuda.current_device())
    print("💠 CUDA device name:", torch.cuda.get_device_name(0))

# Cross Validation
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    print(f"\n🔁 Fold {fold + 1}/{NUM_FOLDS}")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Check for overlap between train and validation sets
    intersect = set(train_df["cleaned"]).intersection(set(val_df["cleaned"]))
    print(f"Overlap between train and val (Fold {fold+1}):", len(intersect))

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    print("📦 Model is on:", next(model.parameters()).device)

    output_dir = os.path.join(RESULT_DIR, f"fold_{fold+1}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        save_total_limit=1,
        seed=SEED,
        metric_for_best_model="eval_f1",
        greater_is_better=True
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    )

    trainer.train()

    # --- Save eval metrics ---
    preds = trainer.predict(val_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    # Confidence analysis
    probs = softmax(preds.predictions, axis=1)
    confidences = np.max(probs, axis=1)
    conf_stats = pd.Series(confidences).describe()
    print("🔍 Confidence stats:\n", conf_stats)
    conf_stats.to_csv(os.path.join(output_dir, "confidence_stats.csv"))

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))
    np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), cm, fmt="%d")

    # --- Plot confusion matrix ---
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Valid", "Hoax"], yticklabels=["Valid", "Hoax"])
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # --- Plot training/validation loss ---
    logs = trainer.state.log_history
    log_df = pd.DataFrame(logs)

    # Plot training & validation loss with filtering
    plt.figure()

    # Ambil hanya row yang punya loss & eval_loss secara terpisah
    train_log = log_df[log_df["loss"].notnull()]
    val_log = log_df[log_df["eval_loss"].notnull()]

    if not train_log.empty:
        plt.plot(train_log["epoch"], train_log["loss"], label="Train Loss", color='blue')

    if not val_log.empty:
        plt.plot(val_log["epoch"], val_log["eval_loss"], label="Val Loss", color='orange')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()


    # --- Store metrics ---
    all_metrics.append({
        "fold": fold + 1,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    })

# --- Summary Report ---
summary_df = pd.DataFrame(all_metrics)
summary_df.to_csv(os.path.join(RESULT_DIR, "summary_metrics.csv"), index=False)
print("\n✅ Training complete. Results saved to", RESULT_DIR)
