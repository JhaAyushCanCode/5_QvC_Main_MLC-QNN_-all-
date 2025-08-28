# prepare_data.py - Enhanced version for GoEmotions dataset preparation
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Paths - UPDATE THESE PATHS TO MATCH YOUR DIRECTORY STRUCTURE
DATA_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs\full_dataset"  # Where your CSV files are
OUTPUT_DIR = r"C:\Users\Admin\.spyder-py3\QvC-3_docs"  # Where to save processed files

# GoEmotions CSV files
files = ["goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"]

# 28 GoEmotions labels in correct order
label_cols = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse','sadness',
    'surprise','neutral'
]

print("🚀 Starting GoEmotions dataset preparation...")
print(f"Expected label columns: {len(label_cols)}")

# Load and combine all CSV files
dfs = []
for f in files:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        print(f"📂 Loading {path}...")
        df = pd.read_csv(path)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        dfs.append(df)
    else:
        print(f"⚠️  Warning: File not found: {path}")

if not dfs:
    print("❌ No CSV files found! Please check your DATA_DIR path.")
    exit(1)

# Combine all dataframes
print("\n🔄 Combining datasets...")
data = pd.concat(dfs, ignore_index=True)
print(f"✅ Combined dataset size: {len(data):,} rows, {len(data.columns)} columns")

# Check for required columns
missing_cols = [col for col in label_cols if col not in data.columns]
if missing_cols:
    print(f"❌ Missing emotion columns: {missing_cols}")
    print(f"Available columns: {list(data.columns)}")
    exit(1)

# Check if 'text' column exists
if 'text' not in data.columns:
    print("❌ 'text' column not found!")
    print(f"Available columns: {list(data.columns)}")
    exit(1)

# Select text + 28 emotions
print(f"🎯 Selecting text column and {len(label_cols)} emotion labels...")
data = data[['text'] + label_cols].copy()

# Check for missing values
print(f"📊 Missing values check:")
print(f"   Text column: {data['text'].isnull().sum()} missing")
print(f"   Label columns: {data[label_cols].isnull().sum().sum()} missing total")

# Remove rows with missing text
initial_size = len(data)
data = data.dropna(subset=['text'])
print(f"   Removed {initial_size - len(data)} rows with missing text")

# Fill missing label values with 0 (assuming binary labels)
data[label_cols] = data[label_cols].fillna(0)

# Ensure label columns are binary (0 or 1)
print("🔢 Converting labels to binary format...")
for col in label_cols:
    unique_vals = data[col].unique()
    print(f"   {col}: unique values = {unique_vals}")
    # Convert to binary if needed
    data[col] = data[col].astype(int)

# Convert label columns into list of ints for each row
print("📋 Converting labels to list format...")
data['labels'] = data[label_cols].apply(lambda row: row.tolist(), axis=1)

# Verify the conversion
sample_labels = data['labels'].iloc[0]
print(f"Sample labels: {sample_labels} (length: {len(sample_labels)})")

# Drop the original individual label columns
data = data[['text', 'labels']].copy()

# Check data quality
print(f"\n📈 Data quality check:")
print(f"   Final dataset size: {len(data):,} rows")
print(f"   Average text length: {data['text'].str.len().mean():.1f} characters")
print(f"   Labels per sample (avg): {np.mean([sum(labels) for labels in data['labels']]):.2f}")

# Split into train/val/test (80/10/10)
print("\n✂️ Splitting dataset...")
train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42, stratify=None)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=None)

print(f"   Train: {len(train_df):,} samples")
print(f"   Val:   {len(val_df):,} samples") 
print(f"   Test:  {len(test_df):,} samples")

# Save the splits
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path   = os.path.join(OUTPUT_DIR, "val.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")

print(f"\n💾 Saving files to {OUTPUT_DIR}...")
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Successfully saved:")
print(f"   📄 {train_path}")
print(f"   📄 {val_path}")
print(f"   📄 {test_path}")

# Verify saved files
print(f"\n🔍 Verification:")
for path, df in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
    if os.path.exists(path):
        loaded = pd.read_csv(path)
        print(f"   ✅ {os.path.basename(path)}: {len(loaded)} rows saved correctly")
    else:
        print(f"   ❌ {os.path.basename(path)}: Failed to save")

# Print label statistics
print(f"\n📊 Label distribution analysis (on training set):")
train_label_stats = []
for labels_list in train_df['labels']:
    # Parse string back to list for analysis
    if isinstance(labels_list, str):
        labels_list = eval(labels_list)
    train_label_stats.extend(labels_list)

train_label_array = np.array(train_label_stats).reshape(-1, len(label_cols))
label_counts = train_label_array.sum(axis=0)

for i, (label, count) in enumerate(zip(label_cols, label_counts)):
    percentage = (count / len(train_df)) * 100
    print(f"   {label:15}: {count:6d} samples ({percentage:5.1f}%)")

print(f"\n🎉 Dataset preparation completed successfully!")
print(f"Ready to train with {len(data):,} total samples!")
