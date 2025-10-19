# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
# resnet_fruits_training.py
# Fine-tune ResNet-50 on Fruits-360 (or similar) using labels CSV metadata.
# - Expects a labels CSV with columns: split (train/test) or separate train/test CSVs
# - Produces checkpoints and prints training/validation metrics
import os
import glob
from pathlib import Path
import pandas as pd
from PIL import Image
import logging
import traceback
from collections import OrderedDict
import types
import __main__
import torch
import sys
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from checkpoint_utils import save_checkpoint, find_latest_checkpoint, load_checkpoint

# ---------------------- Dataset ----------------------
class FruitDataset(Dataset):
    """
    DataFrame expected columns:
      - 'relative_path' or 'filename' (path relative to ROOT_IMAGE_DIR)
      - 'label_id' or 'label_index' (integer class)
      - optionally 'split'
    """
    def __init__(self, df, root_dir, transform=None, path_col="relative_path", label_col="label_id"):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row[self.path_col]
        image_path = self.root_dir.joinpath(rel_path)
        # Robust open
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            if self.transform:
                im = self.transform(im)
        label = int(row[self.label_col])
        return im, label

# ---------------------- Training / Eval functions ----------------------
def train_one_epoch(model, loader, optimizer, criterion, device, logger):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += imgs.size(0)        

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    logger.info(f"Training Epoch finished - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def eval_one_epoch(model, loader, criterion, device, logger):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_corrects += (preds == labels).sum().item()
            total += imgs.size(0)            

    logger.info(f"Eval Epoch finished - Running_loss: {running_loss:.4f}, Running_corrects: {running_corrects:.4f}, Total: {total:.4f}")
    return running_loss / total, running_corrects / total

def main():
    # ---------------------- Logging ----------------------
    log_level = os.getenv("PYTHONLOGLEVEL", "INFO").upper()
    logging.basicConfig(
      level=getattr(logging, log_level),      
      format="%(asctime)s %(levelname)s: %(message)s",
      handlers=[logging.StreamHandler(sys.stdout)],
      force=True  # Only works in Python 3.8+
    )
    logger = logging.getLogger("train")

    # ---------------------- Configuration ----------------------
    logger.info("Configuring")
    '''ROOT_IMAGE_DIR = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/Fruits-360/fruits-360_100x100/fruits-360"
    LABELS_CSV = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier/labels.csv"
    CLASSES_CSV = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier/classes.csv"
    OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier"'''
    ROOT_IMAGE_DIR = "/Users/carloshehe/Desktop/Fruits-360/fruits-360_100x100/fruits-360"
    LABELS_CSV = "/Users/carloshehe/Desktop/Projects/fruit-image-classifier/labels.csv"
    CLASSES_CSV = "/Users/carloshehe/Desktop/Projects/fruit-image-classifier/classes.csv"
    OUTPUT_DIR = "/Users/carloshehe/Desktop/Projects/fruit-image-classifier"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    BATCH_SIZE = 64
    NUM_WORKERS = 4
    NUM_EPOCHS = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    IMAGE_SIZE = 224  # ResNet default

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # ImageNet mean/std (pretrained ResNet expects these)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]    

    # ---------------------- Transforms ----------------------
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # ---------------------- Load CSV and split ----------------------
    df = pd.read_csv(LABELS_CSV)
    logger.info(f"Loaded labels CSV with columns: {df.columns.tolist()} (rows={len(df)})")

    # candidate column names
    path_col_candidates = ['relative_path', 'filename', 'file_name', 'path']
    label_col_candidates = ['label_id', 'label_index', 'label', 'class_index']

    path_col = next((c for c in path_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if path_col is None or label_col is None:
        raise ValueError(f"Could not find path or label column. CSV columns: {df.columns.tolist()}")

    logger.info(f"Using path_col='{path_col}', label_col='{label_col}'")
    logger.debug(f"DataFrame columns before sampling: {df.columns.tolist()}")
    # ---------------------- Optional quick-sampling for tests ----------------------
    # SAMPLE_FRACTION can be set via env var e.g. SAMPLE_FRACTION=0.01 python ...
    try:
        SAMPLE_FRACTION = float(os.environ.get("SAMPLE_FRACTION", "1.0"))
    except Exception:
        SAMPLE_FRACTION = 0.01

    if SAMPLE_FRACTION <= 0 or SAMPLE_FRACTION > 1:
        logger.warning(f"Invalid SAMPLE_FRACTION={SAMPLE_FRACTION}; skipping sampling and using full dataset.")
        SAMPLE_FRACTION = 1.0

    if SAMPLE_FRACTION < 1.0:
        logger.info(f"Sampling a fraction of the dataset for quick tests: SAMPLE_FRACTION={SAMPLE_FRACTION}")                        
        sampled_parts = []
        original_len = len(df)
        for _, group in df.groupby(label_col):
            n = max(1, int(len(group) * SAMPLE_FRACTION))
            sampled_parts.append(group if n >= len(group) else group.sample(n=n, random_state=42))
                
        sampled_df = pd.concat(sampled_parts, ignore_index=True)
        logger.info(f"Sampled dataset rows: {len(sampled_df)} (original {original_len})")

        df = sampled_df   # replace df with sampled_df for downstream splitting

    logger.debug(f"DataFrame columns after sampling: {df.columns.tolist()}")
    # If 'split' exists, normalize and use it; otherwise do stratified split
    if 'split' in df.columns:
        df['split_norm'] = df['split'].astype(str).str.strip().str.lower()
        def map_split(s):
            if s.startswith('train'):
                return 'train'
            if s.startswith('test'):
                return 'test'
            if s in ['val', 'valid', 'validation']:
                return 'val'
            return s
        df['split_norm'] = df['split_norm'].apply(map_split)
        logger.info(f"Split value counts (normalized):\\n{df['split_norm'].value_counts()}")
        if 'train' in df['split_norm'].values or 'test' in df['split_norm'].values or 'val' in df['split_norm'].values:
            train_df = df[df['split_norm'] == 'train'].reset_index(drop=True)
            val_df = df[df['split_norm'].isin(['test', 'val'])].reset_index(drop=True)
            # optional: if there is also explicit 'test' split and it's separate, you can load it here
            test_df = df[df['split_norm'] == 'test'].reset_index(drop=True)
        else:
            logger.warning("split column present but not standard; falling back to stratified split.")
            train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = pd.DataFrame([], columns=df.columns)
    else:
        logger.info("No 'split' column found — performing a stratified train/val split (80/20).")
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = pd.DataFrame([], columns=df.columns)

    # If test_df empty, you can set it equal to val_df or leave it empty - here we keep it empty if not provided.
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    logger.debug(f"DataFrame columns after sampling and split: {df.columns.tolist()}")

    # ---------------------- Build datasets / loaders ----------------------
    train_ds = FruitDataset(train_df, ROOT_IMAGE_DIR, transform=train_transform, path_col=path_col, label_col=label_col)
    val_ds   = FruitDataset(val_df,   ROOT_IMAGE_DIR, transform=val_transform,   path_col=path_col, label_col=label_col)
    test_ds  = FruitDataset(test_df,  ROOT_IMAGE_DIR, transform=val_transform,   path_col=path_col, label_col=label_col) if len(test_df)>0 else None

    # guard
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty. Check LABELS_CSV and 'split' values.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda')) if test_ds is not None else None

    # ---------------------- Model setup ----------------------
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    # num classes
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame columns: {df.columns.tolist()}")
    NUM_CLASSES = int(df[label_col].nunique())
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # ---------------------- Optionally freeze backbone ----------------------
    def set_parameter_requires_grad(model, feature_extracting=True):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

    FEATURE_EXTRACT = True
    set_parameter_requires_grad(model, feature_extracting=FEATURE_EXTRACT)

    # ---------------------- Loss, Optimizer, Scheduler ----------------------
    criterion = nn.CrossEntropyLoss()
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
# ---------------------- Resume / checkpoint setup ----------------------
    start_epoch = 0
    best_val_acc = 0.0
    ckpt = find_latest_checkpoint(OUTPUT_DIR, logger)
    if ckpt:
        try:            
            logger.info(f"Found existing checkpoint: {ckpt}. Resuming training.")
            start_epoch, best_val_acc = load_checkpoint(ckpt, model, logger, optimizer=optimizer, scheduler=scheduler, device=DEVICE)            
            logger.info(f"Resuming from checkpoint. start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {ckpt}: {e}\\nStarting from scratch.")

    # ---------------------- Training loop ----------------------
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            logger.info(f"Starting epoch {epoch} out of {NUM_EPOCHS}")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, logger)

            # Optional quick eval on training set for sanity-check (small overhead)
            train_eval_loss, train_eval_acc = eval_one_epoch(model, train_loader, criterion, DEVICE, logger)

            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE, logger)

            scheduler.step()

            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                        f"eval_train_loss={train_eval_loss:.4f} eval_train_acc={train_eval_acc:.4f}  "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            ckpt_state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_acc": best_val_acc
            }
            # save latest checkpoint always
            save_checkpoint(ckpt_state, os.path.join(OUTPUT_DIR, "latest.pth"), logger)

            # save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(OUTPUT_DIR, f"best_epoch_{epoch+1:03d}_valacc_{val_acc:.4f}.pth")
                save_checkpoint(ckpt_state, best_path, logger)
                logger.info(f"Saved new best model to {best_path} (val_acc={val_acc:.4f})")

    except Exception as e:
        # Save latest on exception to avoid losing progress
        tb = traceback.format_exc()
        logger.error(f"Training failed unexpectedly: {e}\\n{tb}")
        try:
            save_checkpoint({
                "epoch": epoch + 1 if 'epoch' in locals() else 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_acc": best_val_acc
            }, os.path.join(OUTPUT_DIR, "latest_on_error.pth"), logger)
            logger.info("Saved latest checkpoint to latest_on_error.pth")
        except Exception as se:
            logger.error(f"Failed to save checkpoint on error: {se}")
        raise

    # ---------------------- Final evaluation on test set (optional) ----------------------
    # If a best model was saved, load it for test evaluation
    best_ckpt = None
    best_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "best_epoch_*.pth")), key=os.path.getmtime, reverse=True)
    if best_files:
        best_ckpt = best_files[0]
    elif os.path.exists(os.path.join(OUTPUT_DIR, "latest.pth")):
        best_ckpt = os.path.join(OUTPUT_DIR, "latest.pth")

    if best_ckpt:
        logger.info(f"Loading best checkpoint for final evaluation: {best_ckpt}")
        load_checkpoint(best_ckpt, model, logger, device=DEVICE)

    if test_loader is not None:
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, DEVICE, logger)
        logger.info(f"TEST final: loss={test_loss:.4f}, acc={test_acc:.4f}")
    else:
        logger.info("No test split provided - skipping final test evaluation.")

    # ---------------------- Save class map for inference convenience ----------------------
    if os.path.exists(CLASSES_CSV):
        logger.info(f"classes CSV exists at {CLASSES_CSV}")
    else:
        # write small classes CSV mapping if not present
        uniq = sorted(df[label_col].unique())
        classes_df = pd.DataFrame({"class_index": list(range(len(uniq))), "class_name": [str(x) for x in uniq]})
        classes_df.to_csv(CLASSES_CSV, index=False)
        logger.info(f"Wrote fallback classes CSV to {CLASSES_CSV}")

    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
