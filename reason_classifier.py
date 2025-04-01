# core/reason_classifier.py

import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
import joblib 
import os 

# Load data
DATA_PATH = "data/failure_reasons.csv"
df = pd.read_csv(DATA_PATH)
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["reason"])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenization 
MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Dataset class
class ReasonDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings 
        self.labels = labels 
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}
    
# Model init
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label_encoder.classes_))

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_strategy="no",
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ReasonDataset(train_encodings, train_labels),
    eval_dataset=ReasonDataset(val_encodings, val_labels),
    tokenizer=tokenizer
)

# Train and save
trainer.train()
os.makedirs("models/failure_classifier", exist_ok=True)
model.save_pretrained("models/failure_classifier")
tokenizer.save_pretrained("models/failure_classifier")
joblib.dump(label_encoder, "models/failure_classifier/label_encoder.pkl")