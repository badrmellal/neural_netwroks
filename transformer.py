import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math
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


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)
        # Create a vector of shape (max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)

        # Register the positional encoding as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # (batch_size, num_heads, seq_len_q, d_k) x (batch_size, num_heads, d_k, seq_len_k)
        # -> (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask (optional)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # (batch_size, num_heads, seq_len_q, seq_len_k) x (batch_size, num_heads, seq_len_v, d_k)
        # -> (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, d_k)
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x, batch_size):
        # Transpose from (batch_size, num_heads, seq_len, d_k) to (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2).contiguous()
        # Combine the last two dimensions: (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        return x.view(batch_size, -1, self.d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and split heads
        Q = self.split_heads(self.W_q(query), batch_size)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(self.W_k(key), batch_size)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(self.W_v(value), batch_size)  # (batch_size, num_heads, seq_len_v, d_k)

        # Apply scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output projection
        output = self.W_o(self.combine_heads(attn_output, batch_size))

        return output


# Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, max_seq_length, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x


# Transformer Classifier
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_length, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=tokenizer.pad_token_id)
        self.transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, max_seq_length, dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert attention mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None

        # Get embeddings
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]

        # Apply transformer encoder
        encoded = self.transformer_encoder(x, mask)

        # Global average pooling (only considers positions that are not padded)
        if attention_mask is not None:
            # Mask out padding
            encoded = encoded * attention_mask.unsqueeze(-1)
            # Average over sequence length
            pooled = encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            # If no mask, just average all tokens
            pooled = encoded.mean(dim=1)

        # Pass through classifier
        return self.classifier(self.dropout(pooled))


# Hyperparameters
vocab_size = tokenizer.vocab_size
d_model = 128  # Embedding dimension
num_heads = 4  # Number of attention heads
d_ff = 256  # Feed-forward hidden layer dimension
num_layers = 2  # Number of encoder layers
max_seq_length = 512  # Maximum sequence length
num_classes = 4  # Number of classes
dropout_rate = 0.1  # Dropout rate
num_epochs = 5  # Number of training epochs
lr = 5e-4  # Learning rate

model = TransformerClassifier(
    vocab_size,
    d_model,
    num_heads,
    d_ff,
    num_layers,
    max_seq_length,
    num_classes,
    dropout_rate
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    total_steps=num_epochs * len(train_loader)
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(outputs, batch["labels"])
        loss.backward()
        optimizer.step()
        scheduler.step()
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