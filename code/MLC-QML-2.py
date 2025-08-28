# MLC-QML_v2.py
# Quantum-Classical Hybrid Multi-Label Classifier on GoEmotions
# Fixed dataset splits + labels extraction + embedding projection

# Changes vs v1:
#   - Replaced BCELoss with BCEWithLogitsLoss + pos_weight
#   - Added LayerNorm + tanh after projection
#   - Reduced N_LAYERS from 4 → 2
#   - Increased LR from 2e-5 → 5e-4
#   - Added gradient-clipping
#   - Metrics use macro-F1 for early stopping
#   - All other code paths kept identical → no breaking changes.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # suppress tokenizer warnings


#  CONFIG + HYPERPARAMS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE   = 64
MAX_LEN      = 100
EPOCHS       = 35
PATIENCE     = 5
LR           = 5e-4          # << was 2e-5
N_QUBITS     = 8
N_LAYERS     = 2             # << was 4
GRAD_CLIP    = 1.0           # new

print(f"Using device: {device}")
print(f"Hyperparams -> BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, QUBITS={N_QUBITS}, LAYERS={N_LAYERS}, LR={LR}")


# LOAD DATASET

print("Loading GoEmotions dataset...")
raw = load_dataset("go_emotions")

# **** helper: list of indices → multi-hot vector **** 
all_label_names = raw["train"].features["labels"].feature.names
N_LABELS = len(all_label_names)

def row_to_multihot(row):
    vec = np.zeros(N_LABELS, dtype=np.float32)
    for idx in row["labels"]:
        vec[idx] = 1.0
    return vec

#  FULL train split (211 k + 5 k)
full_train_df = pd.concat([
    raw["train"].to_pandas(),
    raw["validation"].to_pandas()
]).reset_index(drop=True)
full_train_df["multi_hot"] = full_train_df.apply(row_to_multihot, axis=1)

# 2) Small 43 k subset for val / test (from original train)
small_df = raw["train"].to_pandas()
small_df["multi_hot"] = small_df.apply(row_to_multihot, axis=1)
_, df_temp      = train_test_split(small_df, test_size=0.2, random_state=42, shuffle=True)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, shuffle=True)

# 3) Final splits
df_train = full_train_df          # 216 k rows
print("Label count:", N_LABELS, "— first 10:", all_label_names[:10])
print("Dataset sizes -> Train:", len(df_train),
      "Val:", len(df_val), "Test:", len(df_test))

# TOKENIZER + BERT  (unchanged)

print("Loading BERT tokenizer + encoder...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

@torch.no_grad()
def encode_texts(texts, max_len=MAX_LEN):
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


#  TORCH DATASETS & LOADERS  (unchanged)

class EmotionDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["text"].tolist()
        self.labels = np.stack(df["multi_hot"].values)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        emb   = encode_texts([self.texts[idx]]).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return emb, label

train_ds = EmotionDataset(df_train)
val_ds   = EmotionDataset(df_val)
test_ds  = EmotionDataset(df_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# QUANTUM CIRCUIT

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")  # << Y rotation
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

class HybridQuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, N_QUBITS)
        self.norm = nn.LayerNorm(N_QUBITS)       
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc1 = nn.Linear(N_QUBITS, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, N_LABELS)

    def forward(self, x):
        x = self.norm(torch.tanh(self.proj(x)))   
        x_cpu = x.detach().cpu()                 
        q_out = self.q_layer(x_cpu).to(x.device)
        x = torch.relu(self.fc1(q_out))
        x = self.dropout(x)
        return self.fc2(x)                       

model = HybridQuantumClassifier().to(device)


#  LOSS + OPTIMIZER 
 
labels_tensor = torch.tensor(train_ds.labels, dtype=torch.float32)

pos_weight = (
    (labels_tensor.shape[0] - labels_tensor.sum(0))
    / labels_tensor.sum(0).clamp(min=1)
).clamp(max=10).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


#  TRAINING LOOP

print("Starting training...")
train_losses, val_macro_f1s = [], []
best_macro, wait = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    #  validation macro-F1 
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    val_macro_f1 = f1_score(all_labels, (all_preds > 0.5).astype(int), average="macro")
    val_macro_f1s.append(val_macro_f1)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {avg_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")

    # Early stopping
    if val_macro_f1 > best_macro:
        best_macro, wait = val_macro_f1, 0
        torch.save(model.state_dict(), "best_model.pt")   # save best checkpoint
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early-stopping at epoch {epoch} (best macro-F1 = {best_macro:.4f})")
            break


#  TEST EVALUATION

print("Evaluating on test set...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

print(classification_report(all_labels,
                            (all_preds > 0.5).astype(int),
                            target_names=all_label_names,
                            zero_division=0))


#  PLOTS

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), val_macro_f1s, marker='o', color='orange')
plt.title("Validation Macro-F1")
plt.xlabel("Epoch"); plt.ylabel("F1")
plt.tight_layout(); plt.show()


# CONFUSION MATRICES + PER-LABEL PERFORMANCE

cm = multilabel_confusion_matrix(all_labels, (all_preds > 0.5).astype(int))
fig, axes = plt.subplots(4, 7, figsize=(25, 15))
axes = axes.ravel()
for i, ax in enumerate(axes):
    if i < N_LABELS:
        sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(all_label_names[i])
        ax.set_xlabel("Pred"); ax.set_ylabel("True")
    else:
        ax.axis("off")
plt.tight_layout(); plt.show()

prec, rec, f1s, supp = precision_recall_fscore_support(
    all_labels, (all_preds > 0.5).astype(int), average=None, zero_division=0)

df_perf = pd.DataFrame({
    "Label": all_label_names,
    "Precision": prec,
    "Recall": rec,
    "F1": f1s,
    "Support": supp
}).sort_values("F1", ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x="F1", y="Label", data=df_perf, palette="viridis")
plt.title("Per-label F1 (test set)")
plt.tight_layout(); plt.show()

print(df_perf)

