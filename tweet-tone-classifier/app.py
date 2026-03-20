# app.py
import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from rewriter import rewrite_tweet

# ── Load from HuggingFace Hub instead of local path ──────
MODEL_NAME = "KinSlay3rs/tweet-tone-classifier"  # <- only change

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = {0: "😠 Negative", 1: "😊 Positive"}

def classify_and_rewrite(tweet, tone):
    if not tweet.strip():
        return "Please enter a tweet.", ""

    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred = logits.argmax().item()
    confidence = torch.softmax(logits, dim=1).max().item()
    label = f"{LABELS[pred]} (confidence: {confidence:.2f})"

    rewritten = rewrite_tweet(tweet, tone)
    return label, rewritten

demo = gr.Interface(
    fn=classify_and_rewrite,
    inputs=[
        gr.Textbox(label="Original Tweet", placeholder="Type a tweet...", lines=3),
        gr.Dropdown(
            choices=["formal", "casual", "empathetic", "assertive"],
            label="Target Tone",
            value="formal"
        )
    ],
    outputs=[
        gr.Textbox(label="Detected Sentiment"),
        gr.Textbox(label="Rewritten Tweet", lines=3)
    ],
    title="🐦 Tweet Tone Classifier & Rewriter",
    description="Detects the sentiment of a tweet and rewrites it in your chosen tone using Gemini AI.",
    examples=[
        ["I can't believe my flight got cancelled again!!", "empathetic"],
        ["Just got promoted!! Best day ever 🎉", "formal"],
        ["This product is absolutely terrible waste of money", "assertive"],
        ["Missing my friends so much today", "casual"],
    ],
    theme=gr.themes.Soft()
)


demo.launch() 

