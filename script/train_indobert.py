import os
import shutil
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from scipy.special import softmax
import wandb

# --- Config ---
MODEL_NAME = "indobenchmark/indobert-base-p1"
DATA_PATH = "cleandata/train_dataset.csv"
RESULT_DIR = "results_indobert"
SAVE_DIR = "saved_model"
NUM_FOLDS = 5
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 2
SEED = 42
WANDB_PROJECT = "IndoBERT-Hoax-Detection"

os.makedirs(RESULT_DIR, exist_ok=True)

# # --- Full Reset ---
# CLEAR_PREVIOUS = True

# if CLEAR_PREVIOUS:
#     # Remove old results
#     if os.path.exists(RESULT_DIR):
#         print(f"🧹 Clearing previous results at '{RESULT_DIR}'...")
#         shutil.rmtree(RESULT_DIR)
#     os.makedirs(RESULT_DIR, exist_ok=True)

#     # Optional: clear wandb cache (local logs)
#     if os.path.exists("wandb"):
#         print("🧹 Clearing local Weights & Biases logs...")
#         shutil.rmtree("wandb")

# --- Init Weights & Biases ---
wandb.init(project=WANDB_PROJECT, name="IndoBERT-5Fold", reinit=True)

# --- Load Dataset ---
df = pd.read_csv(DATA_PATH).dropna(subset=["Text", "label"])
df["label"] = df["label"].astype(int)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["Text"], truncation=True, padding=True, max_length=512)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Using device:", device)

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
all_metrics = []

# --- Training per fold ---
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    print(f"\n🔁 Fold {fold+1}/{NUM_FOLDS}")
    output_dir = os.path.join(RESULT_DIR, f"fold_{fold+1}")
    os.makedirs(output_dir, exist_ok=True)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=1,
        seed=SEED,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to="wandb"
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

    # Save best model for this fold
    # Save model: best if exists, else save last trained 
    model_best = os.path.join(output_dir, "pytorch_model.bin")
    model_final = os.path.join(output_dir, "best_model.bin")

    if os.path.exists(model_best):
        shutil.copyfile(model_best, model_final)
        print(f"✅ [Fold {fold+1}] Best model found and saved as best_model.bin")
    else:
        print(f"⚠️ [Fold {fold+1}] Best model NOT found — saving last model manually...")
        trainer.save_model(output_dir)  # Saves model to output_dir
        # Re-check
        if os.path.exists(model_best):
            shutil.copyfile(model_best, model_final)
            print(f"✅ [Fold {fold+1}] Last model saved as best_model.bin")
        else:
            print(f"❌ [Fold {fold+1}] Failed to save any model! Check training log.")

    # Evaluation on validation set
    preds = trainer.predict(val_dataset)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)
    probs = softmax(preds.predictions, axis=1)
    pd.Series(np.max(probs, axis=1)).describe().to_csv(os.path.join(output_dir, "confidence_stats.csv"))

    pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose().to_csv(
        os.path.join(output_dir, "classification_report.csv"))
    np.savetxt(os.path.join(output_dir, "confusion_matrix.txt"), confusion_matrix(y_true, y_pred), fmt="%d")

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Valid", "Hoax"], yticklabels=["Valid", "Hoax"])
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Loss curve
    log_df = pd.DataFrame(trainer.state.log_history)
    plt.figure()
    if not log_df[log_df["loss"].notnull()].empty:
        plt.plot(log_df["epoch"], log_df["loss"], label="Train Loss", color="blue")
    if not log_df[log_df["eval_loss"].notnull()].empty:
        plt.plot(log_df["epoch"], log_df["eval_loss"], label="Val Loss", color="orange")
    plt.title(f"Loss Curve - Fold {fold+1}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    all_metrics.append({
        "fold": fold + 1,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    })

# Save summary
summary_df = pd.DataFrame(all_metrics)
summary_df.to_csv(os.path.join(RESULT_DIR, "summary_metrics.csv"), index=False)
print("\n✅ Training complete. Results saved to:", RESULT_DIR)

# Save model and tokenizer for deployment
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Final model + tokenizer saved to → {SAVE_DIR}")

# End wandb run
wandb.finish()
