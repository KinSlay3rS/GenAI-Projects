# 🤖 GenAI Projects

A collection of Generative AI and NLP projects built using state-of-the-art 
models, LLMs, and modern AI frameworks.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)
![Gemini](https://img.shields.io/badge/Google-Gemini-green)
![LangChain](https://img.shields.io/badge/🦜-LangChain-teal)
![License](https://img.shields.io/badge/License-Apache%202.0-red)

---

## 👨‍💻 About

This repository contains my hands-on GenAI projects covering fine-tuning,
RAG pipelines, LLM integrations, and AI-powered applications — built as 
part of my journey into Generative AI as a fresher.

Each project is self-contained with its own code, README, and live demo 
where applicable.

---

## 🗂️ Projects

---

### 01 · 🐦 Tweet Tone Classifier & Rewriter

> Detects tweet sentiment and rewrites it in a chosen tone using a 
> fine-tuned DistilBERT model and Google Gemini API.

| | |
|---|---|
| **Type** | Fine-tuning + LLM Integration |
| **Model** | DistilBERT (fine-tuned on Sentiment140) |
| **LLM** | Google Gemini 1.5 Flash |
| **UI** | Gradio |
| **Accuracy** | ~86% on 5,000 test samples |

**Key features:**
- Fine-tuned DistilBERT on 50,000 tweets for binary sentiment classification
- Rewrites tweets in 4 tones — Formal, Casual, Empathetic, Assertive
- Deployed on Hugging Face Spaces with a live demo

**Tech stack:**
`Python` `HuggingFace Transformers` `DistilBERT` `Gemini API` `Gradio` `Sentiment140`

🔗 [View Project](./tweet-tone-classifier) · 
🤗 [Model](https://huggingface.co/KinSlay3rs/tweet-tone-classifier) · 
🚀 [Live Demo](https://huggingface.co/spaces/KinSlay3rs/tweet-tone-classifier)

---

## 🛠️ Tech Stack

### Frameworks & Libraries
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### LLM APIs
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)

### Tools & Platforms
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)
![HuggingFace Spaces](https://img.shields.io/badge/HF%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)

---

## 📁 Repository Structure
```
GenAI-Projects/
│
├── Sentiment-Analysis-DistilBERT/   ← Project 01
│   ├── app.py
│   ├── rewriter.py
│   ├── train.py
│   ├── evaluate.py
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
│
├── project-02/                      ← Coming soon
│
└── README.md                        ← You are here
```

---

## 🚀 Getting Started

Each project has its own `requirements.txt` and `README.md` with setup 
instructions. General steps for any project:

# 1. Clone the repo
git clone https://github.com/KinSlay3rs/GenAI-Projects.git
cd GenAI-Projects

# 2. Navigate to a project
cd Sentiment-Analysis-DistilBERT

# 3. Create a virtual environment
conda create -n project-env python=3.11
conda activate project-env

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your API keys
cp .env.example .env
# edit .env with your keys

# 6. Run
python app.py

---

## 🙋 Author

**KinSlay3rs**

> 🤗 [HuggingFace](https://huggingface.co/KinSlay3rs) ·
> 💻 [GitHub](https://github.com/KinSlay3rS) ·
> 💼 [LinkedIn](https://www.linkedin.com/in/rahul-singh-657b8924b/)

---

## ⭐ Support

If you find these projects helpful, consider giving the repo a star — 
it helps others discover the work and motivates me to keep building!

![Star this repo](https://img.shields.io/github/stars/KinSlay3rs/GenAI-Projects?style=social)
