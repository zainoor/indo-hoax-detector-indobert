# 🇮🇩 Indo-Hoax-Detector-IndoBERT

A hoax detection system for Bahasa Indonesia news articles using IndoBERT (`indobenchmark/indobert-base-p1`). This project fine-tunes IndoBERT on a labeled and balanced dataset compiled from various Indonesian news sources, with evaluation against updated 2025 samples.

---

## 📁 Project Structure

```
.
├── app/                        # Streamlit app interface
├── cleandata/                  # Cleaned + merged labeled datasets
│   ├── hoax_dataset_2025.csv   # Evaluation dataset (2025)
│   └── hoax_dataset_merged.csv # Merged training dataset
├── evaluation/                 # Notebooks for model evaluation
│   ├── evaluate_indobert.ipynb
│   └── evaluate_indobert_kfold.ipynb
├── rawdata/                    # Raw labeled source datasets
│   ├── cnn.xlsx
│   ├── kompas.xlsx
│   ├── tempo.xlsx
│   ├── turnbackhoax.xlsx
│   ├── hoax_valid_labeled.csv
│   └── random_news.csv         # Additional 2025 labeled dataset
├── models/, results/           # Outputs from training and evaluation
├── script/                     # Training and preprocessing scripts
├── .streamlit/                 # Streamlit app config
├── requirement.txt
└── README.md
```

---

## 🧠 Model Info

| Model | IndoBERT                                                                                  |
| ----- | ----------------------------------------------------------------------------------------- |
| Base  | [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1) |
| Task  | Binary classification: `0 = Valid`, `1 = Hoax`                                            |

The model is fine-tuned on curated and labeled datasets, balancing both valid and hoax classes.

---

## 🧪 Training Setup

* Framework: Hugging Face Transformers + `Trainer`
* Cross-Validation: **5-Fold Stratified**
* Token length: 256 tokens
* Optimizer: AdamW
* Epochs: 4
* Batch Size: 16

### 📊 Cross-Validation Results

| Fold | Train Accuracy | Val Accuracy | Val F1 |
| ---- | -------------- | ------------ | ------ |
| 1    | 0.9977         | 0.9905       | 0.9905 |
| 2    | 0.9999         | 0.9887       | 0.9887 |
| 3    | 0.9996         | 0.9903       | 0.9903 |
| 4    | 0.9997         | 0.9900       | 0.9900 |
| 5    | 0.9980         | 0.9897       | 0.9897 |

> 🔹 **Deployed Fold**: **Fold 1**

---

## 📊 Evaluation on 2025 Dataset

Using `hoax_dataset_2025.csv` (464 samples):

```
              precision    recall  f1-score   support

       Valid       0.95      0.92      0.93       203
        Hoax       0.94      0.97      0.95       261

    Accuracy                           0.94       464
   Macro Avg       0.95      0.94      0.94       464
Weighted Avg       0.94      0.94      0.94       464
```

> IndoBERT demonstrates consistent high performance on new 2025 data.

---

## 📂 Labeled Dataset Sources

All datasets are manually labeled prior to training and evaluation.

| File                     | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `cnn.xlsx`               | News articles from CNN Indonesia (from Kaggle dataset)                       |
| `kompas.xlsx`            | News articles from Kompas (from Kaggle dataset)                              |
| `tempo.xlsx`             | News articles from Tempo (from Kaggle dataset)                               |
| `turnbackhoax.xlsx`      | Hoax articles from TurnBackHoax.id (from Kaggle dataset)                     |
| `hoax_valid_labeled.csv` | Mixed-source dataset (manually relabeled: `Hoax`/`Valid` → `0`/`1`)          |
| `random_news.csv`        | Additional news articles (source unspecified; manually labeled `0`/`1`)      |

📦 CNN, Kompas, Tempo, and TurnBackHoax are sourced from the Kaggle dataset:
🔗 [https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news](https://www.kaggle.com/datasets/linkgish/indonesian-fact-and-hoax-political-news)

Cleaned and processed versions are located in the `cleandata/` directory.

---

## 🚀 Running the App

To launch the Streamlit-based news article detector:

```bash
pip install -r requirement.txt
streamlit run app/app.py
```

Paste or input articles and receive real-time predictions.
You can also try the live demo of the app here:
🔗 https://indobert-hoax-detector.streamlit.app/

---

## 📌 TODO

* Expand to multi-class classification: satire, misleading, fabricated, AI-generated, etc.
* Integrate fast summarization for long articles
* Continually enrich the dataset with 2025–2026 samples
* Enable image-based article detection
* Support article input via direct URLs
* Improve the user interface experience
* Incorporate unverified sources (e.g., Twitter, Facebook, WhatsApp)

---

## 💾 Model Hosting (Hugging Face)

Access or load the model directly via:

[https://huggingface.co/zainoor/indo-hoax-detector-indobert](https://huggingface.co/zainoor/indo-hoax-detector-indobert)

No registration required for local use. Hosted for inference or integration.

---
