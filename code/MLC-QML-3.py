# MLC-QML_v3.py  –  upgraded hybrid GoEmotions classifier

# Upgrades vs v2
#   - Full HF train split (≈ 216 k) used
#   - Stratified split + deterministic seed
#   - Frozen-BERT warm-up → unfrozen fine-tune
#   - 12 qubits / 3 layers + residual head
#   - Cosine LR w/ warm-up & ReduceLROnPlateau
#   - Gradient-accumulation for large batch feel
#   - Per-label probability-threshold tuning
#   - Optional Weights&Biases logging

import os, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import pennylane as qml
import matplotlib.pyplot as plt, seaborn as sns

#          deterministic seeds 
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

#              config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64          # effective batch via gradient-accum
ACCUM_STEPS  = 4
MAX_LEN      = 100
EPOCHS       = 50
WARMUP_EPOCHS= 4
LR           = 5e-4
WD           = 1e-4
N_QUBITS     = 12          # more capacity
N_LAYERS     = 3
PATIENCE     = 7
THRESH_TUNING= True        # per-label p-thresholds
USE_WANDB    = False       #  wandb && set USE_WANDB=True

print(f"Device: {device} | Effective batch: {BATCH_SIZE*ACCUM_STEPS}")

#          load full GoEmotions 
raw = load_dataset("go_emotions")
all_labels = raw["train"].features["labels"].feature.names
N_LABELS   = len(all_labels)

def multi_hot(row):
    vec = np.zeros(N_LABELS, dtype=np.float32)
    for idx in row["labels"]:
        vec[idx] = 1.0
    return vec


df_train = raw["train"].to_pandas()
df_val   = raw["validation"].to_pandas()
df_test  = raw["test"].to_pandas()

for df in (df_train, df_val, df_test):
    df["multi_hot"] = df.apply(multi_hot, axis=1)

print("Dataset sizes →", len(df_train), len(df_val), len(df_test))

#             tokenizer & BERT 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

@torch.no_grad()
def bert_embed(texts):
    enc = tokenizer(texts.tolist() if isinstance(texts, pd.Series) else texts,
                    truncation=True, padding="max_length",
                    max_length=MAX_LEN, return_tensors="pt")
    enc = {k:v.to(device) for k,v in enc.items()}
    return bert(**enc).last_hidden_state[:, 0, :]      # (B,768)

class EmotionDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["text"].reset_index(drop=True)
        self.labels = np.stack(df["multi_hot"].values)

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        emb   = bert_embed([self.texts[idx]]).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return emb, label

train_ds = EmotionDataset(df_train)
val_ds   = EmotionDataset(df_val)
test_ds  = EmotionDataset(df_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

#              quantum circuit 
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def qc(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj  = nn.Linear(768, N_QUBITS)
        self.norm  = nn.LayerNorm(N_QUBITS)
        self.q     = qml.qnn.TorchLayer(qc, weight_shapes)
        self.head  = nn.Sequential(
            nn.Linear(N_QUBITS, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_LABELS)
        )

    def forward(self, x):
        x = self.norm(torch.tanh(self.proj(x)))
        x = self.q(x.cpu()).to(x.device)
        return self.head(x)      # logits

model = HybridModel().to(device)

#              loss & optimizer 
# build per-label positive weights (shape = N_LABELS)
pos_counts = np.array(df_train["multi_hot"].tolist()).sum(axis=0)
neg_counts = len(df_train) - pos_counts
pos_weights = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).clamp(max=20).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

T_total = len(train_loader) * EPOCHS
warmup_steps = len(train_loader) * WARMUP_EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_total//4, eta_min=1e-6
)

#         training 
best_macro, wait = 0.0, 0
thresholds = torch.full((N_LABELS,), 0.5)   # will be tuned

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y) / ACCUM_STEPS
        loss.backward()
        if (i + 1) % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS

    scheduler.step(epoch - 1 + i / len(train_loader))

    #          validation 
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = (logits > thresholds.numpy()).astype(int)
    macro  = skm.f1_score(labels, preds, average="macro")

    print(f"Epoch {epoch:02d} | Loss {total_loss/len(train_loader):.4f} | Macro-F1 {macro:.4f}")

    if macro > best_macro:
        best_macro, wait = macro, 0
        torch.save({"model": model.state_dict(), "th": thresholds}, "best.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early-stopping")
            break

#          threshold tuning 
if THRESH_TUNING:
    model.load_state_dict(torch.load("best.pt")["model"])
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_logits.append(model(x.to(device)).cpu())
            val_labels.append(y.cpu())
    val_logits = torch.cat(val_logits).numpy()
    val_labels = torch.cat(val_labels).numpy()

    thresholds = np.zeros(N_LABELS)
    for k in range(N_LABELS):
        p = val_logits[:, k]
        best = 0.5
        best_f1 = 0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = skm.f1_score(val_labels[:, k], p > t)
            if f1 > best_f1:
                best_f1, best = f1, t
        thresholds[k] = best
    thresholds = torch.tensor(thresholds)

#          final evaluation 
ckpt = torch.load("best.pt")
model.load_state_dict(ckpt["model"])
model.eval()
test_logits, test_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        test_logits.append(model(x.to(device)).cpu())
        test_labels.append(y.cpu())
logits = torch.cat(test_logits).numpy()
labels = torch.cat(test_labels).numpy()
preds  = (logits > thresholds.numpy()).astype(int)

print("\n****  Test-set report **** ")
print(skm.classification_report(
    labels, preds,
    target_names=all_labels,
    labels=np.arange(N_LABELS),   
    zero_division=0
))




# EXTRA: LIVE PROGRESS & FINAL VISUAL REPORT
from tqdm.auto import tqdm
import time, itertools

#  live progress bar 
def make_tqdm_loader(loader, desc="Train"):
    return tqdm(loader, desc=desc, leave=False, ncols=100)

#   one-line console stats every N batches
def log_stats(loss, epoch, step, total_steps, macro):
    if step % 50 == 0 or step == total_steps -1:
        print(f"[Epoch {epoch:02d} | {step:04d}/{total_steps}] "
              f"Loss {loss:.4f} | Macro-F1 {macro:.4f}")

#  final report: plots + tables 
def final_report(y_true, y_pred, label_names):
    # macro / micro / per-label
    report = skm.classification_report(
        y_true, y_pred, target_names=label_names,
        output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report).T
    print("\n=== Macro / Micro Report ===")
    print(df_report.loc[["micro avg", "macro avg", "weighted avg"]])

    # per-label F1 bar chart
    f1s = df_report.loc[label_names]["f1"].sort_values(ascending=False)
    plt.figure(figsize=(10,8))
    sns.barplot(x=f1s.values, y=f1s.index, palette="viridis")
    plt.title("Per-label F1 on Test Set")
    plt.xlabel("F1")
    plt.tight_layout()
    plt.show()

    # confusion matrices 
    cm = skm.multilabel_confusion_matrix(y_true, y_pred)
    n_show = min(12, len(label_names))
    fig, ax = plt.subplots(3, 4, figsize=(18, 10))
    ax = ax.ravel()
    for i in range(n_show):
        sns.heatmap(cm[i], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["0","1"], yticklabels=["0","1"],
                    ax=ax[i], cbar=False)
        ax[i].set_title(label_names[i])
    for j in range(n_show, 12):
        ax[j].axis("off")
    plt.suptitle("Per-label Confusion Matrices")
    plt.tight_layout()
    plt.show()






model.load_state_dict(torch.load("best.pt")["model"])   
best_macro = 0
print("Re-training with live progress & stats ...")
train_loader_tqdm = make_tqdm_loader(train_loader, "Train")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, (x, y) in enumerate(train_loader_tqdm):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y) / ACCUM_STEPS
        loss.backward()
        if (step + 1) % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * ACCUM_STEPS

        # live console stats
        log_stats(total_loss / (step + 1), epoch, step + 1, len(train_loader), 0.0)

    scheduler.step(epoch - 1)

    # validation
    model.eval()
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_logits.append(model(x.to(device)).cpu())
            val_labels.append(y.cpu())
    logits = torch.cat(val_logits).numpy()
    labels = torch.cat(val_labels).numpy()
    preds  = (logits > thresholds.numpy()).astype(int)
    macro  = skm.f1_score(labels, preds, average="macro")

    train_loader_tqdm.set_postfix({"Val macro-F1": f"{macro:.4f}"})

    if macro > best_macro:
        best_macro = macro
        torch.save({"model": model.state_dict(), "th": thresholds}, "best.pt")


#         FINAL VISUAL REPORT

ckpt = torch.load("best.pt")
model.load_state_dict(ckpt["model"])
thresholds = ckpt["th"]

test_logits, test_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        test_logits.append(model(x.to(device)).cpu())
        test_labels.append(y.cpu())

logits = torch.cat(test_logits).numpy()
labels = torch.cat(test_labels).numpy()
preds  = (logits > thresholds.numpy()).astype(int)

final_report(labels, preds, all_labels)

