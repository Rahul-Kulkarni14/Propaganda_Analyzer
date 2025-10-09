# =============================
# performance.py - Evaluate Trained Models (Binary + Multi-class)
# =============================
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # Suppress TF messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Disable oneDNN notices
warnings.filterwarnings("ignore")             # Suppress Python warnings
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

torch.backends.cudnn.benchmark = False        # Avoid extra cuDNN logs
# =============================
# Section 0: Check GPU
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# =============================
# Section 1: Load Dataset
# =============================
df = pd.read_csv('processed/final_dataset.csv')  # Same dataset as main.py

# Binary labels
y_bin = df['label']
X_bin = df['text_fragment']

# Multi-class labels
df['technique_label'] = df['technique'].factorize()[0]
y_mc = df['technique_label']
X_mc = df['text_fragment']

# =============================
# Section 2: Load Tokenizer
# =============================
tokenizer = DistilBertTokenizerFast.from_pretrained('transformer_binary_model')

max_length = 128

# =============================
# Section 3: Tokenize Data
# =============================
enc_bin = tokenizer(list(X_bin), truncation=True, padding=True, max_length=max_length)
enc_mc = tokenizer(list(X_mc), truncation=True, padding=True, max_length=max_length)

# =============================
# Section 4: Dataset Class
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

dataset_bin = PropagandaDataset(enc_bin, y_bin)
dataset_mc = PropagandaDataset(enc_mc, y_mc)

loader_bin = DataLoader(dataset_bin, batch_size=12, shuffle=False)
loader_mc = DataLoader(dataset_mc, batch_size=12, shuffle=False)

# =============================
# Section 5: Load Trained Models
# =============================
model_bin = DistilBertForSequenceClassification.from_pretrained('transformer_binary_model').to(device)
model_mc = DistilBertForSequenceClassification.from_pretrained('transformer_multiclass_model').to(device)

# =============================
# Section 6: Evaluation Function
# =============================
def evaluate_model(model, loader, task_name="Model"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(f"\n===== {task_name} Performance =====")
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# =============================
# Section 7: Evaluate Binary Model
# =============================
evaluate_model(model_bin, loader_bin, task_name="Binary Model")

# =============================
# Section 8: Evaluate Multi-class Model
# =============================
evaluate_model(model_mc, loader_mc, task_name="Multi-class Model")
