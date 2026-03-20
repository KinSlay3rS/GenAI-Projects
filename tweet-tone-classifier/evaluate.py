# evaluate.py
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import pandas as pd

# ── Load model and tokenizer ──────────────────────────────
MODEL_PATH = "./tweet-classifier"  # or "KinSlay3rs/tweet-tone-classifier"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Running on: {device}")

# ── Load test data ────────────────────────────────────────
# load the same dataset used during training
df = pd.read_csv("dataset/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
df.columns = ["label", "id", "date", "query", "user", "text"]
df["label"] = df["label"].apply(lambda x: 1 if x == 4 else 0)

# use the same 50k sample and same random state as training
df = df.sample(50000, random_state=42)

# take 10% as test set (same split as training)
test_df = df.sample(frac=0.1, random_state=42)
print(f"Test samples: {len(test_df)}")

# ── Prediction function ───────────────────────────────────
def get_predictions(texts, batch_size=64):
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=64
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # progress
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

    return np.array(all_preds), np.array(all_probs)

# ── Run predictions ───────────────────────────────────────
texts = test_df["text"].tolist()
true_labels = test_df["label"].tolist()

print("Running predictions...")
predictions, probabilities = get_predictions(texts)

# ── Compute metrics ───────────────────────────────────────
accuracy  = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall    = recall_score(true_labels, predictions)
f1        = f1_score(true_labels, predictions)

print("\n" + "="*50)
print("         MODEL EVALUATION RESULTS")
print("="*50)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
print("="*50)

# ── Full classification report ────────────────────────────
print("\nDetailed Classification Report:")
print(classification_report(
    true_labels,
    predictions,
    target_names=["Negative", "Positive"]
))

# ── Confusion matrix ──────────────────────────────────────
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(f"                 Predicted")
print(f"                 Neg    Pos")
print(f"Actual  Neg  |  {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"        Pos  |  {cm[1][0]:5d}  {cm[1][1]:5d}")