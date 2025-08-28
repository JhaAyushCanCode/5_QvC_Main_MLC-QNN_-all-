# MLC-QML_v4.py — upgraded hybrid GoEmotions classifier (Modified for local dataset)

import os, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
import pennylane as qml
import matplotlib.pyplot as plt, seaborn as sns
from tqdm.auto import tqdm
import time, itertools
import ast  # For parsing string representations of lists

#                deterministic seeds 
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

#                    config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64          
ACCUM_STEPS  = 4
MAX_LEN      = 100
EPOCHS       = 500
WARMUP_EPOCHS= 4
LR           = 8e-4
WD           = 1e-4
N_QUBITS     = 12         
N_LAYERS     = 3
PATIENCE     = 189
THRESH_TUNING= True        
USE_WANDB    = False       

# Dataset paths 
DATA_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs"  # *********************** Update required before rerun
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH   = os.path.join(DATA_DIR, "val.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

print(f"Device: {device} | Effective batch: {BATCH_SIZE*ACCUM_STEPS}")

#            load local dataset files

# Define emotion labels 
all_labels = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse','sadness',
    'surprise','neutral'
]

N_LABELS = len(all_labels)
print(f"Found {N_LABELS} emotion labels: {all_labels[:10]}...")

def parse_labels(label_str):
    """Parse string representation of labels list"""
    if isinstance(label_str, str):
        try:
            return ast.literal_eval(label_str)
        except:
            # Fallback parsing if ast.literal_eval fails
            return eval(label_str)
    return label_str

def labels_to_multi_hot(labels_list):
    """Convert list of binary labels to multi-hot vector"""
    vec = np.array(labels_list, dtype=np.float32)
    return vec

# Load the prepared datasets
print("Loading prepared datasets...")
try:
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    
    # Parse labels from string format
    df_train["labels"] = df_train["labels"].apply(parse_labels)
    df_val["labels"]   = df_val["labels"].apply(parse_labels)
    df_test["labels"]  = df_test["labels"].apply(parse_labels)
    
    # Convert to multi-hot format
    df_train["multi_hot"] = df_train["labels"].apply(labels_to_multi_hot)
    df_val["multi_hot"]   = df_val["labels"].apply(labels_to_multi_hot)
    df_test["multi_hot"]  = df_test["labels"].apply(labels_to_multi_hot)
    
    print("Dataset sizes →", len(df_train), len(df_val), len(df_test))
    
except FileNotFoundError as e:
    print(f"Error: Could not find dataset files. Please run the data preparation pipeline first.")
    print(f"Expected files: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")
    print(f"Error details: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)

#                tokenizer & BERT 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

# Pre-compute all embeddings 
print("Pre-computing BERT embeddings .  .   .  .")

@torch.no_grad()
def bert_embed_batch(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        embeddings = bert(**enc).last_hidden_state[:, 0, :]  # (B,768)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)

# Pre-compute embeddings
train_embeddings = bert_embed_batch(df_train["text"].tolist())
val_embeddings = bert_embed_batch(df_val["text"].tolist())
test_embeddings = bert_embed_batch(df_test["text"].tolist())

class EmotionDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = np.stack(labels)

    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx):
        return self.embeddings[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

train_ds = EmotionDataset(train_embeddings, df_train["multi_hot"].values)
val_ds   = EmotionDataset(val_embeddings, df_val["multi_hot"].values)
test_ds  = EmotionDataset(test_embeddings, df_test["multi_hot"].values)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

#                quantum circuit 
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
        x = x.to(device)  # Ensure input is on correct device
        x = self.norm(torch.tanh(self.proj(x)))
        # Handle quantum circuit computation
        x_quantum = self.q(x.cpu()).to(device)  # Quantum ops on CPU, then back to GPU
        return self.head(x_quantum)      # logits

model = HybridModel().to(device)

#                loss & optimizer 
# build per-label positive weights (shape = N_LABELS)
pos_counts = np.array(df_train["multi_hot"].tolist()).sum(axis=0)
neg_counts = len(df_train) - pos_counts
pos_weights = torch.tensor(neg_counts / np.maximum(pos_counts, 1), dtype=torch.float32).clamp(max=20).to(device)

print(f"Positive weights shape: {pos_weights.shape}")
print(f"Sample positive weights: {pos_weights[:5]}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

T_total = len(train_loader) * EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=T_total//4, eta_min=1e-6
)

#                helper functions 
def log_stats(loss, epoch, step, total_steps, lr):
    if step % 50 == 0 or step == total_steps - 1:
        print(f"[Epoch {epoch:02d} | {step:04d}/{total_steps}] "
              f"Loss {loss:.4f} | LR {lr:.6f}")

def evaluate_model(model, loader, thresholds):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x).cpu().numpy())  # Convert to numpy immediately
            all_labels.append(y.cpu().numpy())  # Convert to numpy immediately
    
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    #   pakka PAKKA thresholds is numpy array
    if isinstance(thresholds, torch.Tensor):
        thresh_np = thresholds.numpy()
    else:
        thresh_np = thresholds
        
    preds = (logits > thresh_np).astype(int)
    macro = skm.f1_score(labels, preds, average="macro", zero_division=0)
    return macro, logits, labels, preds

#                training 
best_macro, wait = 0.0, 0
thresholds = torch.full((N_LABELS,), 0.5)   # will be tuned

print("Starting training -  -  -  -- -")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Progress bar for batches
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
    
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y) / ACCUM_STEPS
        loss.backward()
        
        if (i + 1) % ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * ACCUM_STEPS
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{total_loss/(i+1):.4f}",
            'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
        })

    scheduler.step()

    #                    validation 
    macro, _, _, _ = evaluate_model(model, val_loader, thresholds)
    
    print(f"Epoch {epoch:02d} | Loss {total_loss/len(train_loader):.4f} | "
          f"Val Macro-F1 {macro:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}")

    if macro > best_macro:
        best_macro, wait = macro, 0
        torch.save({
            "model": model.state_dict(), 
            "th": thresholds,
            "epoch": epoch,
            "macro": macro
        }, "best.pt", _use_new_zipfile_serialization=False)
        print(f"New best model saved! Macro-F1: {macro:.4f}")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

#                threshold tuning 
print("Loading best model for threshold tuning...")
ckpt = torch.load("best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])

if THRESH_TUNING:
    print("Tuning per-label thresholds...")
    _, val_logits, val_labels, _ = evaluate_model(model, val_loader, thresholds)

    thresholds = np.zeros(N_LABELS)
    for k in tqdm(range(N_LABELS), desc="Tuning thresholds"):
        p = val_logits[:, k]
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 81):
            f1 = skm.f1_score(val_labels[:, k], p > t, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[k] = best_t
    
    thresholds = torch.tensor(thresholds)
    print("Threshold tuning completed!")

#                final evaluation 
print("Final evaluation on test set...")
test_macro, test_logits, test_labels, test_preds = evaluate_model(model, test_loader, thresholds)

print(f"\n==== Final Test Results ====")
print(f"Test Macro-F1: {test_macro:.4f}")

print("\n==== Detailed Classification Report ====")
try:
    # Ensure correct number of labels
    if len(all_labels) != N_LABELS:
        print(f"Warning: Label count mismatch. Using first {N_LABELS} labels.")
        display_labels = all_labels[:N_LABELS]
    else:
        display_labels = all_labels
    
    # Convert tensors to numpy arrays
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()
    if isinstance(test_preds, torch.Tensor):
        test_preds = test_preds.numpy()
    
    report = skm.classification_report(
        test_labels, test_preds,
        target_names=display_labels,
        labels=np.arange(N_LABELS),   
        zero_division=0
    )
    print(report)
except Exception as e:
    print(f"Error generating classification report: {e}")
    print("Continuing with basic metrics...")
    # Basic metrics without detailed report
    macro_f1 = skm.f1_score(test_labels, test_preds, average='macro', zero_division=0)
    micro_f1 = skm.f1_score(test_labels, test_preds, average='micro', zero_division=0)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")

#            visualization functions
def final_report(y_true, y_pred, label_names):
    """Generate comprehensive visual report"""
    # Classification report as DataFrame
    report = skm.classification_report(
        y_true, y_pred, target_names=label_names,
        output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report).T
    
    print("\n=== Summary Metrics ===")
    summary = df_report.loc[["micro avg", "macro avg", "weighted avg"]]
    print(summary)

    # Per-label F1 scores visualization
    label_f1s = df_report.loc[label_names]["f1-score"].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(label_f1s)), label_f1s.values)
    plt.yticks(range(len(label_f1s)), label_f1s.index)
    plt.xlabel("F1 Score")
    plt.title("Per-Label F1 Scores on Test Set")
    plt.grid(axis='x', alpha=0.3)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if label_f1s.values[i] > 0.5:
            bar.set_color('green')
        elif label_f1s.values[i] > 0.3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.show()
    
    return df_report

#     final report
try:
    df_report = final_report(test_labels, test_preds, all_labels)
    
    # Save results
    results = {
        'test_macro_f1': test_macro,
        'model_params': sum(p.numel() for p in model.parameters()),
        'best_epoch': ckpt.get('epoch', 'unknown'),
        'dataset_sizes': {
            'train': len(df_train),
            'val': len(df_val),
            'test': len(df_test)
        },
        'config': {
            'n_qubits': N_QUBITS,
            'n_layers': N_LAYERS,
            'batch_size': BATCH_SIZE,
            'lr': LR
        }
    }
    
    # Save detailed results (anxiety, not compulsory)
    pd.DataFrame([results]).to_csv('experiment_results.csv', index=False)
    torch.save({
        'final_model': model.state_dict(),
        'thresholds': thresholds,
        'results': results,
        'test_predictions': test_preds,
        'test_labels': test_labels
    }, 'final_results.pt', _use_new_zipfile_serialization=False)
    
    print(f"\nResults saved! Final test macro-F1: {test_macro:.4f}")
    print(f"Model parameters: {results['model_params']:,}")
    print(f"Dataset sizes - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
except Exception as e:
    print(f"Error in final report generation: {e}")
    print("But training completed successfully!")

print("\n=== Training Complete ===")
print(f"Best validation macro-F1: {best_macro:.4f}")
print(f"Final test macro-F1: {test_macro:.4f}")

