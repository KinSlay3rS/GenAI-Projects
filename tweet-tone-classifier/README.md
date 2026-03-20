# 🐦 Tweet Tone Classifier & Rewriter

A GenAI-powered NLP application that detects the sentiment of a tweet and rewrites 
it in a chosen tone using a fine-tuned DistilBERT model and Google Gemini API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow)
![Gemini](https://img.shields.io/badge/Google-Gemini%20API-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-red)

---

## 📌 Overview

This project has two core features:

- **Sentiment Classification** — Fine-tuned DistilBERT model trained on 50,000 tweets 
  from the Sentiment140 dataset. Predicts whether a tweet is Positive or Negative with 
  ~86% accuracy.
- **Tone Rewriting** — Uses Google Gemini API to rewrite any tweet in four different 
  tones: Formal, Casual, Empathetic, and Assertive.

---

## 🎯 Demo

> 🔗 Live Demo: [huggingface.co/spaces/KinSlay3rs/tweet-tone-classifier](https://huggingface.co/spaces/KinSlay3rs/tweet-tone-classifier)  
> 🤗 Model: [huggingface.co/KinSlay3rs/tweet-tone-classifier](https://huggingface.co/KinSlay3rs/tweet-tone-classifier)

---

## 🏗️ Project Structure
```
Sentiment-Analysis-DistilBERT/
├── app.py                                    ← Gradio UI
├── rewriter.py                               ← Gemini API tone rewriter
├── distilbert-sentiment-analysis.ipynb       ← Fine-tuning script
├── evaluate.py                               ← Evaluation metrics script
├── requirements.txt                          ← Dependencies
├── .env.example                              ← Environment variable template
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Classification Model | DistilBERT (fine-tuned) |
| Training Framework | HuggingFace Transformers |
| LLM | Google Gemini 1.5 Flash |
| LLM Integration | Google GenerativeAI SDK |
| UI | Gradio |
| Dataset | Sentiment140 (50k samples) |
| Model Hosting | HuggingFace Hub |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~86% |
| Precision | ~87% |
| Recall | ~85% |
| F1 Score | ~86% |

> Evaluated on 5,000 held-out test samples from Sentiment140.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Google Gemini API key — get it free at [aistudio.google.com](https://aistudio.google.com)

### Installation

# 1. Clone the repo
git clone https://github.com/KinSlay3rs/tweet-tone-classifier.git
cd tweet-tone-classifier

# 2. Create and activate virtual environment
conda create -n tweet-classifier python=3.11
conda activate tweet-classifier

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# then open .env and add your Gemini API key


### Environment Variables

Create a `.env` file in the root directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Run the app

python app.py

Then open your browser at `http://127.0.0.1:7860`

---

## 🧠 How It Works
```
Input Tweet
     │
     ▼
Preprocessing (remove URLs, handles, special chars)
     │
     ├──────────────────────┬─────────────────────────┐
     ▼                      ▼                         │
DistilBERT             Gemini API                     │
Classifier             Rewriter                       │
     │                      │                         │
     ▼                      ▼                         │
Sentiment Label        Rewritten Tweet                │
(Positive/Negative)    (in chosen tone)               │
     │                      │                         │
     └──────────────────────┘                         │
                    │                                  │
                    ▼                                  │
              Gradio UI  ◄────────────────────────────┘
```

---

## 🏋️ Training

The model was trained on Kaggle using a T4 GPU.

If you want to retrain the model yourself:

# 1. Download Sentiment140 dataset from Kaggle
# https://www.kaggle.com/datasets/kazanova/sentiment140

# 2. Place it in the project root as training.1600000.processed.csv

# 3. Run training
python train.py


**Training config:**
- Base model: `distilbert-base-uncased`
- Dataset: 50,000 samples from Sentiment140
- Epochs: 3
- Batch size: 32
- Max token length: 64
- Training time: ~25 mins on T4 GPU

---

## 📈 Evaluation

Run the evaluation script to get full metrics:

python evaluate.py

Output:
```
==================================================
         MODEL EVALUATION RESULTS
==================================================
  Accuracy  : 0.8623  (86.23%)
  Precision : 0.8701  (87.01%)
  Recall    : 0.8544  (85.44%)
  F1 Score  : 0.8622  (86.22%)
==================================================
```

---

## 🎨 Supported Tones

| Tone | Description |
|---|---|
| **Formal** | Professional and corporate language |
| **Casual** | Friendly and conversational |
| **Empathetic** | Warm and understanding |
| **Assertive** | Confident and direct |

---

## ⚠️ Limitations

- English tweets only — may not generalize to other languages
- Sarcasm and irony are often misclassified
- Trained on 2009 tweets — modern slang may reduce accuracy
- Binary classification only — no neutral class

---

## 🔭 Future Work

- [ ] Add neutral sentiment class
- [ ] Support multilingual tweets using `xlm-roberta-base`
- [ ] Add emoji-aware preprocessing
- [ ] Fine-tune on more recent tweet data
- [ ] Add confidence threshold — only rewrite if confidence > 0.8

---

## 📜 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 🙋 Author

**KinSlay3rs**

> 🤗 [HuggingFace](https://huggingface.co/KinSlay3rs) · 
> 💻 [GitHub](https://github.com/KinSlay3rs)

---

## 🙏 Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) by Stanford
- [DistilBERT](https://huggingface.co/distilbert-base-uncased) by HuggingFace
- [Google Gemini API](https://aistudio.google.com)
- [Gradio](https://gradio.app) for the UI framework
```

---

**Also create a `.env.example` file** — this goes on GitHub (unlike `.env`) so others know what variables are needed:
```
# get your free Gemini API key at https://aistudio.google.com
GEMINI_API_KEY=your_gemini_api_key_here
