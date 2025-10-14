# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% id="UwgMRoz1LeiP" executionInfo={"status": "ok", "timestamp": 1759705047834, "user_tz": 240, "elapsed": 103, "user": {"displayName": "Carlos", "userId": "17576890038933523144"}}

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1759705088441, "user_tz": 240, "elapsed": 40589, "user": {"displayName": "Carlos", "userId": "17576890038933523144"}} outputId="fcd750d0-8ca2-4179-f82e-a67d8d30eed9" id="xzq68SQBPX-I"
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier

# %% colab={"base_uri": "https://localhost:8080/"} id="jvJM2tdVPZxR" executionInfo={"status": "ok", "timestamp": 1759705373920, "user_tz": 240, "elapsed": 151482, "user": {"displayName": "Carlos", "userId": "17576890038933523144"}} outputId="b0dcab2e-c95b-4564-a303-b9078f13c922"
import os
import re
import pandas as pd

# --- Configuration ---
# Root folder that contains "Training" and "Test" subfolders
root_dir = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/Fruits-360/fruits-360_100x100/fruits-360"
labels_csv = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/Fruits-360/labels.csv"
classes_csv = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/Fruits-360/classes.csv"

# --- Helper function to clean and normalize class names ---
def normalize_label(name):
    """
    Normalize folder names to consistent class labels.
    - Removes digits and hyphens (e.g., 'tomato 9' → 'tomato')
    - Replaces spaces/underscores with a single underscore
    - Converts to lowercase
    """
    name = re.sub(r'[\d\-]+', '', name)           # remove digits and hyphens
    name = re.sub(r'[\s_]+', '_', name.strip())   # collapse multiple spaces/underscores
    return name.lower()

# --- Step 1: Identify both splits ---
splits = ["Training", "Test"]
records = []
class_names = set()

for split in splits:
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        print(f"⚠️ Skipping missing split folder: {split_dir}")
        continue

    for folder_name in os.listdir(split_dir):
        class_path = os.path.join(split_dir, folder_name)
        print(f"On class path: {class_path}")
        if not os.path.isdir(class_path):
            continue

        label_name = normalize_label(folder_name)
        class_names.add(label_name)

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                records.append({
                    'split': split.lower(),
                    'filename': os.path.join(split, folder_name, file_name),  # relative path
                    'label_name': label_name
                })

# --- Step 2: Build class index mapping ---
unique_classes = sorted(class_names)
label_to_index = {name: idx for idx, name in enumerate(unique_classes)}

# Add numeric index to records
for rec in records:
    rec['label_index'] = label_to_index[rec['label_name']]

# --- Step 3: Create DataFrames and Save ---
labels_df = pd.DataFrame(records)
classes_df = pd.DataFrame(list(label_to_index.items()), columns=['class_name', 'class_index'])

labels_df.to_csv(labels_csv, index=False)
classes_df.to_csv(classes_csv, index=False)

print(f"✅ Saved {len(labels_df)} image labels to {labels_csv}")
print(f"✅ Saved {len(classes_df)} class mappings to {classes_csv}")

# Optional preview
print(labels_df.head())

