"""
Week 5 - Deep Learning Application in Data Science
Project: Handwritten Digit Classification using PyTorch
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Load public dataset
digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)

# 2. Split into train/validation/test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

# 3. Define neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = DigitClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 4. Train with early stopping
best_val_loss = float("inf")
best_state = None
wait = 0
patience = 10

for epoch in range(1, 101):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * len(yb)
            total += len(yb)
    val_loss /= total

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        break

model.load_state_dict(best_state)

# 5. Evaluate on the test set
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test))
    predictions = logits.argmax(1).numpy()

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, average="weighted"))
print("Recall:", recall_score(y_test, predictions, average="weighted"))
print("F1-score:", f1_score(y_test, predictions, average="weighted"))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, digits=4))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))
