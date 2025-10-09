# =============================
# main.py - Propaganda Detection with Transformers (Binary + Multi-class)
# =============================

# =============================
# Section 0: Imports
# =============================
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# =============================
# Section 1: Check GPU
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# =============================
# Section 2: Load Dataset
# =============================
df = pd.read_csv('processed/final_dataset.csv')
print(f"✅ Loaded dataset: {df.shape[0]} samples")
print(df.head())

# =============================
# Section 3: Multi-class labels
# =============================
# Binary: label column is already 0/1
# Multi-class: technique -> integer labels
df['technique_label'] = df['technique'].factorize()[0]
num_classes = df['technique_label'].nunique()
print(f"✅ Number of techniques (multi-class labels): {num_classes}")

# =============================
# Section 4: Train-Test Split
# =============================
# Binary
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    df['text_fragment'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Multi-class
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    df['text_fragment'], df['technique_label'], test_size=0.2, random_state=42, stratify=df['technique_label']
)

# =============================
# Section 5: Tokenization
# =============================
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
max_length = 128

train_enc_bin = tokenizer(list(X_train_bin), truncation=True, padding=True, max_length=max_length)
test_enc_bin = tokenizer(list(X_test_bin), truncation=True, padding=True, max_length=max_length)

train_enc_mc = tokenizer(list(X_train_mc), truncation=True, padding=True, max_length=max_length)
test_enc_mc = tokenizer(list(X_test_mc), truncation=True, padding=True, max_length=max_length)

# =============================
# Section 6: Dataset Class
# =============================
class PropagandaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

train_dataset_bin = PropagandaDataset(train_enc_bin, y_train_bin)
test_dataset_bin = PropagandaDataset(test_enc_bin, y_test_bin)

train_dataset_mc = PropagandaDataset(train_enc_mc, y_train_mc)
test_dataset_mc = PropagandaDataset(test_enc_mc, y_test_mc)

# =============================
# Section 7: DataLoaders
# =============================
train_loader_bin = DataLoader(train_dataset_bin, batch_size=12, shuffle=True)
test_loader_bin = DataLoader(test_dataset_bin, batch_size=12, shuffle=False)

train_loader_mc = DataLoader(train_dataset_mc, batch_size=12, shuffle=True)
test_loader_mc = DataLoader(test_dataset_mc, batch_size=12, shuffle=False)

# =============================
# Section 8: Model Initialization
# =============================
# Binary
model_bin = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model_bin.to(device)

# Multi-class
model_mc = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_classes)
model_mc.to(device)

# =============================
# Section 9: Optimizers
# =============================
optimizer_bin = AdamW(model_bin.parameters(), lr=5e-5)
optimizer_mc = AdamW(model_mc.parameters(), lr=5e-5)

# =============================
# Section 10: Training Function
# =============================
from tqdm import tqdm

def train_model(model, optimizer, train_loader, epochs=3):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_description(f"Loss {loss.item():.4f}")

# =============================
# Section 11: Evaluation Function
# =============================
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# =============================
# Section 12: Train & Evaluate Binary Model
# =============================
print("\n===== Training Binary Model =====")
train_model(model_bin, optimizer_bin, train_loader_bin, epochs=3)
print("\n✅ Binary Model Evaluation:")
evaluate_model(model_bin, test_loader_bin)

# =============================
# Section 13: Train & Evaluate Multi-class Model
# =============================
print("\n===== Training Multi-class Model =====")
train_model(model_mc, optimizer_mc, train_loader_mc, epochs=3)
print("\n✅ Multi-class Model Evaluation:")
evaluate_model(model_mc, test_loader_mc)

# =============================
# Section 14: Save Models & Tokenizer
# =============================
# Binary
bin_dir = 'transformer_binary_model'
os.makedirs(bin_dir, exist_ok=True)
model_bin.save_pretrained(bin_dir)
tokenizer.save_pretrained(bin_dir)

# Multi-class
mc_dir = 'transformer_multiclass_model'
os.makedirs(mc_dir, exist_ok=True)
model_mc.save_pretrained(mc_dir)
tokenizer.save_pretrained(mc_dir)

print("\n✅ All models and tokenizer saved successfully")
