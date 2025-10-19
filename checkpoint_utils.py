import os
import glob
import torch
import logging
from collections import OrderedDict
import types
import __main__

def save_checkpoint(state, path, logger):
    """Saves a checkpoint to the given path."""
    logger.info(f"Saving checkpoint to: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def find_latest_checkpoint(folder, logger):
    """Finds the latest .pth checkpoint in a folder."""
    logger.info(f"Searching for latest checkpoint in folder: {folder}")
    latest = os.path.join(folder, "latest.pth")
    if os.path.exists(latest):
        return latest
    
    files = glob.glob(os.path.join(folder, "*.pth"))
    if not files:
        logger.info("No checkpoint files found.")
        return None
    
    # Sort by modification time in descending order
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return files[0]

def load_checkpoint(path, model, logger, optimizer=None, scheduler=None, device=torch.device("cpu")):
    """Loads a checkpoint and configures the model, optimizer, and scheduler."""
    # Ensure a dummy logger exists to allow unpickling if the training script used one
    logger.info(f"Loading checkpoint from: {path}")
    if not hasattr(__main__, "logger"):
        __main__.logger = types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            debug=lambda *a, **k: None
        )

    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        print(f"Warning: torch.load failed ({e}), retrying with dummy logger...")
        # This fallback can help if the checkpoint was saved in an environment with a logger
        __main__.logger = types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            debug=lambda *a, **k: None
        )
        ckpt = torch.load(path, map_location=device)

    # Extract state dict from checkpoint, handling different checkpoint formats
    state = ckpt.get("model_state_dict", ckpt)

    # Strip 'module.' prefix if the model was saved using DataParallel
    new_state = OrderedDict()
    for k, v in state.items():
        name = k[len("module."):] if k.startswith("module.") else k
        new_state[name] = v

    # Load weights into the model
    model.load_state_dict(new_state, strict=False)

    # Optionally load optimizer state
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"Could not load optimizer state: {e}")

    # Optionally load scheduler state
    if scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"Could not load scheduler state: {e}")

    start_epoch = ckpt.get("epoch", 0)
    best_val_acc = ckpt.get("best_val_acc", 0.0)
    
    return start_epoch, best_val_acc
