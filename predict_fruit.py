#!/usr/bin/env python3
"""Run a single-image prediction using a saved ResNet-50 checkpoint."""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from checkpoint_utils import load_checkpoint

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_checkpoint(repo: Path) -> Path:
    best = list(repo.glob("best_epoch_*.pth"))
    if best:
        return max(best, key=lambda p: p.stat().st_mtime)
    latest = repo / "latest.pth"
    if latest.is_file():
        return latest
    raise FileNotFoundError(
        f"No best_epoch_*.pth or latest.pth under {repo}. Pass --checkpoint explicitly."
    )


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("predict")

    repo = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Predict fruit class from one image.")
    p.add_argument("image", type=Path, help="Path to an image file (jpg/png/webp/…).")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pth checkpoint (default: newest best_epoch_*.pth in repo).",
    )
    p.add_argument(
        "--classes",
        type=Path,
        default=repo / "classes.csv",
        help="CSV with class_index and class_name (default: ./classes.csv).",
    )
    p.add_argument("--topk", type=int, default=5, help="Number of top predictions to print.")
    args = p.parse_args()

    ckpt_path = args.checkpoint or default_checkpoint(repo)
    ckpt_path = ckpt_path.resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    img_path = args.image.resolve()
    if not img_path.is_file():
        raise FileNotFoundError(img_path)

    classes_csv = args.classes.resolve()
    if not classes_csv.is_file():
        raise FileNotFoundError(classes_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Checkpoint: %s", ckpt_path)

    try:
        peek = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        peek = torch.load(ckpt_path, map_location="cpu")
    state = peek.get("model_state_dict", peek)
    n_cls = state["fc.weight"].shape[0]

    classes_df = pd.read_csv(classes_csv)
    if "class_index" not in classes_df.columns or "class_name" not in classes_df.columns:
        raise ValueError(f"Expected class_index and class_name in {classes_csv}")
    id2name = dict(zip(classes_df["class_index"].astype(int), classes_df["class_name"].astype(str)))

    model = build_model(n_cls, device)
    load_checkpoint(str(ckpt_path), model, logger, device=device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    with Image.open(img_path) as im:
        tensor = tfm(im.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(args.topk, n_cls)
    p_top, idx_top = torch.topk(probs, topk)

    logger.info("Image: %s", img_path)
    for rank, (prob, idx) in enumerate(zip(p_top.tolist(), idx_top.tolist()), start=1):
        name = id2name.get(idx, f"<unknown index {idx}>")
        logger.info("  %d. %s  (p=%.4f, index=%d)", rank, name, prob, idx)


if __name__ == "__main__":
    main()
