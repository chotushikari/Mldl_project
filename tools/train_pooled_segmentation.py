"""Train a two-channel disc/cup U-Net across RIM-ONE, ORIGA, and REFUGE."""
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
EPOCHS = 8
SIZE = 256
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = Path("outputs/pooled_segmentation")
root.mkdir(parents=True, exist_ok=True)
combined = zipfile.ZipFile("data/raw/ORIGA_candidate_figshare.zip")
rim = zipfile.ZipFile("data/raw/RIM-ONE-r3.zip")

records = []

rim_manifest = {
    row["image"]: row
    for row in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))
}
rim_split = list(csv.DictReader(Path("outputs/rim_unet_baseline/split.csv").open(encoding="utf-8")))
for row in rim_split:
    records.append({
        "dataset": "RIM-ONE r3",
        "image": row["image"],
        "split": row["split"],
    })

origa = list(csv.DictReader(Path("data/audit/origa_manifest.csv").open(encoding="utf-8")))
origa_a = [row for row in origa if row["provided_set"] == "A"]
origa_b = [row for row in origa if row["provided_set"] == "B"]
train_a, val_a = train_test_split(
    np.arange(len(origa_a)),
    test_size=0.2,
    stratify=[row["label"] for row in origa_a],
    random_state=SEED,
)
for i, row in enumerate(origa_a):
    records.append({
        "dataset": "ORIGA",
        "image": row["image"],
        "mask": row["mask"],
        "split": "val" if i in set(val_a) else "train",
    })
for row in origa_b:
    records.append({
        "dataset": "ORIGA",
        "image": row["image"],
        "mask": row["mask"],
        "split": "test",
    })

refuge = list(csv.DictReader(Path("data/audit/refuge_manifest.csv").open(encoding="utf-8")))
train_rows = [row for row in refuge if row["split"] == "train"]
train_labels = [int(row["label"]) for row in train_rows]
train_idx, val_idx = train_split = train_test_split(
    np.arange(len(train_rows)),
    test_size=0.2,
    stratify=train_labels,
    random_state=SEED,
)
for i, row in enumerate(train_rows):
    records.append({
        "dataset": "REFUGE",
        "image": row["image"],
        "mask": row["mask"],
        "split": "val" if i in set(val_idx) else "train",
    })
for row in (r for r in refuge if r["split"] in ("val", "test")):
    records.append({
        "dataset": "REFUGE",
        "image": row["image"],
        "mask": row["mask"],
        "split": "test",
    })

assert len(records) == 159 + 650 + 1200
assert len({(row["dataset"], row["image"]) for row in records}) == len(records)
with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["dataset", "image", "mask", "split"])
    writer.writeheader()
    writer.writerows({key: row.get(key, "") for key in ("dataset", "image", "mask", "split")} for row in records)


class PooledSegmentation(Dataset):
    def __init__(self, indices: list[int], augment: bool = False):
        self.indices = indices
        self.augment = augment
        self.data = []
        for i in indices:
            row = records[i]
            if row["dataset"] == "RIM-ONE r3":
                rim_row = rim_manifest[row["image"]]
                with Image.open(io.BytesIO(rim.read(row["image"]))).convert("RGB") as image:
                    image = image.crop((0, 0, image.width // 2, image.height)).resize(
                        (SIZE, SIZE), Image.Resampling.BICUBIC
                    )
                masks = []
                for key in ("disc_mask", "cup_mask"):
                    with Image.open(io.BytesIO(rim.read(rim_row[key]))).convert("L") as mask:
                        mask = mask.crop((0, 0, mask.width // 2, mask.height)).resize(
                            (SIZE, SIZE), Image.Resampling.NEAREST
                        )
                        masks.append(np.asarray(mask) > 127)
                truth = np.stack(masks).astype("float32")
            else:
                with Image.open(io.BytesIO(combined.read(row["image"]))).convert("RGB") as image:
                    image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
                with Image.open(io.BytesIO(combined.read(row["mask"]))).convert("L") as mask:
                    mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
                    values = np.asarray(mask)
                    if row["dataset"] == "REFUGE" and set(np.unique(values)).issubset({0, 1, 2}):
                        pass
                truth = np.stack((values > 0, values == 2)).astype("float32")
            pixels = np.asarray(image).copy()
            self.data.append((pixels, truth))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        pixels, truth = self.data[i]
        pixels = pixels.copy()
        truth = truth.copy()
        if self.augment and random.random() < 0.5:
            pixels = pixels[:, ::-1].copy()
            truth = truth[:, :, ::-1].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float() / 255,
            torch.from_numpy(truth),
        )


train = [i for i, row in enumerate(records) if row["split"] == "train"]
val = [i for i, row in enumerate(records) if row["split"] == "val"]
test = [i for i, row in enumerate(records) if row["split"] == "test"]

train_ds = PooledSegmentation(train, augment=True)
val_ds = PooledSegmentation(val)
test_ds = PooledSegmentation(test)

model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()
history = []
best = -1.0


def scores(model: nn.Module, dataset: Dataset) -> np.ndarray:
    model.eval()
    totals = np.zeros((2, 2), dtype=np.float64)
    count = 0
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=4):
            pred = (model(x.to(device)).sigmoid() > 0.5).cpu().numpy()
            truth = y.numpy() > 0
            for channel in range(2):
                p = pred[:, channel]
                g = truth[:, channel]
                intersection = (p & g).sum(axis=(1, 2))
                predicted = p.sum(axis=(1, 2))
                actual = g.sum(axis=(1, 2))
                union = (p | g).sum(axis=(1, 2))
                totals[channel, 0] += ((2 * intersection + 1e-6) / (predicted + actual + 1e-6)).sum()
                totals[channel, 1] += ((intersection + 1e-6) / (union + 1e-6)).sum()
            count += len(x)
    return totals / count


for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for x, y in DataLoader(train_ds, batch_size=4, shuffle=True):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        prob = logits.sigmoid()
        dice = 1 - (
            (2 * (prob * y).sum((0, 2, 3)) + 1)
            / ((prob + y).sum((0, 2, 3)) + 1)
        ).mean()
        loss = bce(logits, y) + dice
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    val_scores = scores(model, val_ds)
    rank = float(val_scores[:, 0].mean())
    history.append({
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        "val_disc_dice": float(val_scores[0, 0]),
        "val_cup_dice": float(val_scores[1, 0]),
    })
    if rank > best:
        best = rank
        torch.save(model.state_dict(), root / "best.pt")
    print(history[-1], flush=True)

model.load_state_dict(torch.load(root / "best.pt", map_location=device, weights_only=True))
with (root / "history.json").open("w", encoding="utf-8") as stream:
    json.dump(history, stream, indent=2)

test_rows = []
test_indices_by_dataset = {}
for dataset in ("RIM-ONE r3", "ORIGA", "REFUGE"):
    indices = [i for i in test if records[i]["dataset"] == dataset]
    test_indices_by_dataset[dataset] = indices
    subset = PooledSegmentation(indices)
    score = scores(model, subset)
    test_rows.append({
        "dataset": dataset,
        "n_test": len(indices),
        "test_disc_dice": float(score[0, 0]),
        "test_disc_iou": float(score[0, 1]),
        "test_cup_dice": float(score[1, 0]),
        "test_cup_iou": float(score[1, 1]),
    })

counts = {
    split: {
        "n": sum(row["split"] == split for row in records),
        "by_dataset": {
            dataset: sum(row["split"] == split and row["dataset"] == dataset for row in records)
            for dataset in ("RIM-ONE r3", "ORIGA", "REFUGE")
        },
    }
    for split in ("train", "val", "test")
}
summary = {
    "status": "three_dataset_partial_segmentation",
    "architecture": "compact randomly initialized UNet",
    "device": str(device),
    "size": SIZE,
    "seed": SEED,
    "epochs": EPOCHS,
    "datasets": ["RIM-ONE r3", "ORIGA", "REFUGE"],
    "protocol": "RIM-ONE existing 103/24/32; ORIGA A 260/65, B 325 test; REFUGE supplied train 320/80 reallocated, supplied validation and official test 800 test",
    "split_counts": counts,
    "test_metrics": test_rows,
    "limitations": "DRISHTI-GS unavailable; randomly initialized compact U-Net; REFUGE/ORIGA from the verified repackaging; no patient overlap audit for ORIGA.",
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)

combined.close()
rim.close()
