# MLC-QML.py
# Quantum-Classical Hybrid Multi-Label Classifier on GoEmotions
# Fixed dataset splits + labels extraction + embedding projection

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pennylane as qml


#  CONFIG + HYPERPARAMS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
MAX_LEN = 100
EPOCHS = 10
LR = 2e-5
N_QUBITS = 8
N_LAYERS = 4
N_LABELS = 28  

print(f"Using device: {device}")
print(f"Hyperparams -> BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, QUBITS={N_QUBITS}, LAYERS={N_LAYERS}")


# LOAD DATASET

print("Loading GoEmotions dataset...")
raw = load_dataset("go_emotions")

# Convert train split to DataFrame
df_all = raw["train"].to_pandas()

feature_dict = raw["train"].features
print("Dataset features:", feature_dict)

# Extract label names from the ClassLabel list
all_label_names = feature_dict["labels"].feature.names
N_LABELS = len(all_label_names)

# Convert each row’s "labels" list of indices → multi-hot vector
def row_to_multihot(row, label_names=all_label_names):
    vec = np.zeros(len(label_names), dtype=np.float32)
    for idx in row["labels"]:   # row["labels"] is a list of int indices
        vec[idx] = 1.0
    return vec

df_all["multi_hot"] = df_all.apply(lambda row: row_to_multihot(row), axis=1)

# Train / val / test split: 80 / 10 / 10
df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True)
df_val, df_test   = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

print("Using", N_LABELS, "emotion labels ->", all_label_names[:10], "...")
print("Dataset sizes -> Train:", len(df_train), " Val:", len(df_val), " Test:", len(df_test))


#  TOKENIZER + BERT

print("Loading BERT tokenizer + encoder...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()  

@torch.no_grad()
def encode_texts(texts, max_len=MAX_LEN):
    """Encode a small batch/list/Series of texts -> CLS embeddings (B, 768)."""
    if isinstance(texts, (list, tuple)):
        batch_texts = list(texts)
    elif isinstance(texts, pd.Series):
        batch_texts = texts.tolist()
    else:
        batch_texts = [str(texts)]

    enc = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    outputs = bert(
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device)
    )
    return outputs.last_hidden_state[:, 0, :]  # (B, 768)


#  TORCH DATASETS & LOADERS

class EmotionDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        # stack pre-built multi-hot label vectors
        self.labels = np.stack(df["multi_hot"].values)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Encode a single text as a 1-item batch, then squeeze back to (768,)
        emb = encode_texts([self.texts[idx]]).squeeze(0)      # on CUDA if available
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # CPU
        return emb, label

train_ds = EmotionDataset(df_train)
val_ds   = EmotionDataset(df_val)
test_ds  = EmotionDataset(df_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


#  QUANTUM CIRCUIT

# Use CPU default simulator for the quantum part
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # inputs shape per sample: (N_QUBITS,)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

class HybridQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Project BERT (768) -> N_QUBITS
        self.proj = nn.Linear(768, N_QUBITS)

        # Quantum layer (runs on CPU)
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Small head
        self.fc1 = nn.Linear(N_QUBITS, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, N_LABELS)

    def forward(self, x):
        # x is typically on CUDA
        x = self.proj(x)                            # (B, 768) -> (B, N_QUBITS)

        # PennyLane default.qubit expects CPU float32 tensors
        x_cpu = x.detach().to("cpu", dtype=torch.float32)

        q_out_cpu = self.q_layer(x_cpu)             # (B, N_QUBITS) on CPU
        q_out = q_out_cpu.to(x.device, dtype=x.dtype)

        x = torch.relu(self.fc1(q_out))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))           # probabilities for BCE

model = HybridQuantumClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)


#  TRAINING LOOP

print("Starting training...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for inputs, labels in train_loader:
       
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)                 # (B, N_LABELS)
        loss = criterion(outputs, labels)       # BCE on probabilities
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader))

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            preds = model(inputs).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels)         

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    val_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average="micro")

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")


#  TEST EVALUATION

print("Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        preds = model(inputs).detach().cpu()
        all_preds.append(preds)
        all_labels.append(labels)  # CPU

all_preds = torch.cat(all_preds, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

print(classification_report(
    all_labels,
    (all_preds > 0.5).astype(int),
    target_names=all_label_names,
    zero_division=0
))



#  METRICS TRACKING

import matplotlib.pyplot as plt

train_losses, val_f1s = [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    val_f1 = f1_score(all_labels.numpy(), (all_preds.numpy() > 0.5).astype(int), average="micro")
    val_f1s.append(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

# Plot metrics
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), val_f1s, marker='o')
plt.title("Validation F1 (micro)")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.show()




# CONFUSION MATRIX

from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns

cm = multilabel_confusion_matrix(all_labels.numpy(), (all_preds.numpy() > 0.5).astype(int))

fig, axes = plt.subplots(4, 7, figsize=(25, 15))  # 28 emotions
axes = axes.ravel()

for i, ax in enumerate(axes):
    if i < N_LABELS:
        sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(all_label_names[i])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    else:
        ax.axis("off")

plt.tight_layout()
plt.show()




#  PER-LABEL PERFORMANCE

from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1s, support = precision_recall_fscore_support(
    all_labels.numpy(),
    (all_preds.numpy() > 0.5).astype(int),
    average=None
)

df_perf = pd.DataFrame({
    "Label": all_label_names,
    "Precision": prec,
    "Recall": rec,
    "F1": f1s,
    "Support": support
}).sort_values("F1", ascending=False)

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x="F1", y="Label", data=df_perf, palette="viridis")
plt.title("Per-label F1 scores (sorted)")
plt.show()

print(df_perf)

