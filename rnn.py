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


# RNN Model definition
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, bidirectional=True,
                 rnn_type='lstm', dropout_rate=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.dropout = nn.Dropout(dropout_rate)

        # RNN type selection
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        else:  # vanilla RNN
            self.rnn = nn.RNN(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate if num_layers > 1 else 0
            )

        # The output dimension will be doubled if using bidirectional RNN
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embedded = self.dropout(embedded)

        # Pack the sequence to handle variable length inputs efficiently
        # This is not strictly necessary with the current implementation but is good practice
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(
        #     embedded, attention_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False
        # )

        # Pass through RNN
        if self.rnn_type in ['lstm', 'gru']:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        # Different ways to get the final representation:

        # Option 1: Use the last hidden state
        if self.rnn_type == 'lstm':
            # For LSTM, hidden is a tuple (hidden_state, cell_state)
            # We only want the hidden_state
            hidden_state = hidden[0]
        else:
            # For GRU and RNN, hidden is just the hidden state
            hidden_state = hidden

        # Get the correct hidden representation based on directional setting
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            last_hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        else:
            last_hidden = hidden_state[-1, :, :]

        # Option 2: Alternative approach - use attention mask to get the last valid output for each sequence
        # lengths = attention_mask.sum(dim=1).cpu().numpy()
        # batch_size = input_ids.size(0)
        # last_outputs = []
        # for i in range(batch_size):
        #     last_outputs.append(output[i, lengths[i]-1, :])
        # last_output = torch.stack(last_outputs)

        # Pass through final classifier
        return self.fc(last_hidden)


# Hyperparameters
vocab_size = tokenizer.vocab_size
embed_dim = 128
hidden_dim = 64
num_classes = 4
num_epochs = 5
lr = 1e-3
num_layers = 2
bidirectional = True
rnn_type = 'lstm'  # Options: 'lstm', 'gru', 'rnn'

model = RNNClassifier(
    vocab_size,
    embed_dim,
    hidden_dim,
    num_classes,
    num_layers=num_layers,
    bidirectional=bidirectional,
    rnn_type=rnn_type
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["labels"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

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