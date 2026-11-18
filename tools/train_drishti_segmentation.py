"""Train a compact DRISHTI-GS1 Kaggle-mirror optic disc/cup segmentation baseline."""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet


SEED = 42
SIZE = 256
EPOCHS = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

archive = zipfile.ZipFile("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip")
rows = list(csv.DictReader(Path("data/audit/drishti_gs1_kaggle_mirror_manifest.csv").open(encoding="utf-8")))
train_rows = [row for row in rows if row["official_style_split"] == "train"]
test_rows = [row for row in rows if row["official_style_split"] == "test"]
train_idx, val_idx = train_test_split(
    np.arange(len(train_rows)),
    test_size=0.2,
    stratify=[row["label"] for row in train_rows],
    random_state=SEED,
)
train_pool = train_rows
train_rows = [train_pool[i] for i in train_idx]
val_rows = [train_pool[i] for i in val_idx]

root = Path("outputs/drishti_gs1_mirror")
root.mkdir(parents=True, exist_ok=True)
split_rows = (
    [{"image": row["image"], "label": row["label"], "split": "train"} for row in train_rows]
    + [{"image": row["image"], "label": row["label"], "split": "val"} for row in val_rows]
    + [{"image": row["image"], "label": row["label"], "split": "test"} for row in test_rows]
)
with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["image", "label", "split"])
    writer.writeheader()
    writer.writerows(split_rows)


class Drishti(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        with Image.open(io.BytesIO(archive.read(row["image"]))).convert("RGB") as image:
            image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        masks = []
        for key in ("disc_mask", "cup_mask"):
            with Image.open(io.BytesIO(archive.read(row[key]))).convert("L") as mask:
                mask = mask.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                masks.append((np.asarray(mask) >= 128).astype("float32"))
        pixels = np.asarray(image).copy()
        truth = np.stack(masks)
        if self.augment and random.random() < 0.5:
            pixels = pixels[:, ::-1].copy()
            truth = truth[:, :, ::-1].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float() / 255,
            torch.from_numpy(truth),
        )


train_ds = Drishti(train_rows, augment=True)
val_ds = Drishti(val_rows)
test_ds = Drishti(test_rows)
model = UNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
history = []
best = -1.0


def scores(dataset):
    model.eval()
    totals = np.zeros((2, 2), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=4):
            pred = (model(x).sigmoid() > 0.5).numpy()
            truth = y.numpy() > 0
            for channel in range(2):
                p = pred[:, channel]
                g = truth[:, channel]
                inter = (p & g).sum(axis=(1, 2))
                ps = p.sum(axis=(1, 2))
                gs = g.sum(axis=(1, 2))
                union = (p | g).sum(axis=(1, 2))
                totals[channel, 0] += ((2 * inter + 1e-6) / (ps + gs + 1e-6)).sum()
                totals[channel, 1] += ((inter + 1e-6) / (union + 1e-6)).sum()
            count += len(x)
    return totals / count


for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for x, y in DataLoader(train_ds, batch_size=4, shuffle=True):
        logits = model(x)
        prob = logits.sigmoid()
        dice = 1 - ((2 * (prob * y).sum((0, 2, 3)) + 1) / ((prob + y).sum((0, 2, 3)) + 1)).mean()
        loss = bce(logits, y) + dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    val = scores(val_ds)
    rank = float(val[:, 0].mean())
    history.append({
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        "val_disc_dice": float(val[0, 0]),
        "val_cup_dice": float(val[1, 0]),
    })
    if rank > best:
        best = rank
        torch.save(model.state_dict(), root / "best.pt")
    print(history[-1], flush=True)

model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
test = scores(test_ds)
metrics = {
    "status": "drishti_gs1_kaggle_mirror_partial_segmentation",
    "dataset": "DRISHTI-GS1",
    "source": "lokeshsaipureddi Kaggle mirror",
    "architecture": "compact randomly initialized UNet",
    "seed": SEED,
    "epochs": EPOCHS,
    "n_train": len(train_rows),
    "n_val": len(val_rows),
    "n_test": len(test_rows),
    "test_disc_dice": float(test[0, 0]),
    "test_disc_iou": float(test[0, 1]),
    "test_cup_dice": float(test[1, 0]),
    "test_cup_iou": float(test[1, 1]),
    "limitations": "Third-party Kaggle mirror; test soft masks used only for held-out evaluation; official registration not used.",
}
(root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2), flush=True)
archive.close()
