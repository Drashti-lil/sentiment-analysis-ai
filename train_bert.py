# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# # Load dataset
# df = pd.read_csv("final_improved_dataset.csv")

# # Encode labels
# label_map = {"negative":0, "neutral":1, "positive":2}
# df['label'] = df['label'].map(label_map)

# # Split
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42
# )

# # Load tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Tokenize
# train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
# val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# # Dataset class
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels.iloc[idx])
#         return item

#     def __len__(self):
#         return len(self.labels)

# train_dataset = Dataset(train_encodings, train_labels)
# val_dataset = Dataset(val_encodings, val_labels)

# # Load model
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     logging_dir='./logs'
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train
# trainer.train()

# # Save model
# model.save_pretrained("bert_model")
# tokenizer.save_pretrained("bert_model")

# print("✅ BERT model trained!")


import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("final_improved_dataset.csv")

# Encode labels
label_map = {"negative":0, "neutral":1, "positive":2}
df['label'] = df['label'].map(label_map)

# -----------------------------
# Train-test split (IMPORTANT)
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']   # 🔥 important
)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# -----------------------------
# Dataset class
# -----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# -----------------------------
# Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3
)

# -----------------------------
# Metrics (VERY IMPORTANT)
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# -----------------------------
# Training arguments (OPTIMIZED)
# -----------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Evaluate
# -----------------------------
results = trainer.evaluate()
print("\n📊 Evaluation Results:")
print(results)

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("\n✅ BERT model trained & saved!")