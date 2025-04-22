import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AG dataset from huggingface
dataset = load_dataset("ag_news")

# Transformer from huggingface
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(col):
    return tokenizer(col["text"], truncation=True, padding=False)

# To apply the tokenizer to the entire dataset in batches.
tokenized_dataset = dataset.map(tokenize, batched=True)

split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = split_dataset["train"]
val_data = split_dataset["test"]

def collate_fn(batch):
        # turning lists of tensors into padded tensors
        input_ids = pad_sequence([torch.tensor(x["input_ids"]) for x in batch], batch_first=True).to(device)
        attention_mask = pad_sequence([torch.tensor(x["attention_mask"]) for x in batch], batch_first=True).to(device)
        # stack labels into one tensor
        labels = torch.stack([torch.tensor(x["label"]) for x in batch]).to(device)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels
        }

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

# Model def
class DNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(DNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hyperparameters
vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 64
num_classes = 4
num_epochs = 5
lr = 1e-3

model = DNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch["input_ids"], batch["attention_mask"])
            pred = torch.argmax(outputs, dim=1)
            preds.extend(pred.cpu().numpy())
            true.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(true, preds)
    print(f"Validation Accuracy: {acc:.4f}")