# core/inference.py

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import os

# Load model and tokenizer from local path
MODEL_DIR = os.path.abspath("models/failure_classifier")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Ensure these files exist
assert os.path.exists(os.path.join(MODEL_DIR, "label_encoder.pkl")), "Missing model weights!"
assert os.path.exists(LABEL_ENCODER_PATH), "Missing label encoder!"

model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Predict reason from user feedback

def predict_reason(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return label_encoder.inverse_transform([predicted_class])[0]

if __name__ == "__main__":
    feedback = "I was too tired to focus."
    print("Predicted reason:", predict_reason(feedback))
