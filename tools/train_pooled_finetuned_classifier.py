"""Fine-tune a pooled glaucoma ResNet-18 across ACRIMA, RIM-ONE, ORIGA, REFUGE, and DRISHTI-GS1."""
from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Normalize, ToTensor


SEED = 42
EPOCHS = 12
BATCH_SIZE = 32
SIZE = 224
LR = 3e-4
WEIGHT_DECAY = 1e-4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

root = Path("outputs/pooled_finetuned_classifier")
root.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

records = []


def add(dataset, member, label, split, archive_kind="acrima", drishti_label=None, source_split=""):
    records.append({
        "dataset": dataset,
        "member": member,
        "label": int(label),
        "split": split,
        "archive": archive_kind,
        "drishti_label": drishti_label,
        "source_split": source_split,
    })


for row in csv.DictReader(Path("outputs/acrima_resnet18_frozen/split.csv").open(encoding="utf-8")):
    add("ACRIMA", row["member"], row["label"] == "glaucoma", row["split"], "acrima")

for row in csv.DictReader(Path("outputs/rim_unet_baseline/split.csv").open(encoding="utf-8")):
    rim_labels = {r["image"]: r["label_original"] for r in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))}
    label = rim_labels[row["image"]]
    if label in ("normal", "glaucoma"):
        add("RIM-ONE r3", row["image"], label == "glaucoma", row["split"], "rim")

origa = list(csv.DictReader(Path("data/audit/origa_manifest.csv").open(encoding="utf-8")))
origa_a = [row for row in origa if row["provided_set"] == "A"]
origa_b = [row for row in origa if row["provided_set"] == "B"]
origa_train_idx, origa_val_idx = train_test_split(
    np.arange(len(origa_a)),
    test_size=0.2,
    stratify=[row["label"] for row in origa_a],
    random_state=SEED,
)
for i, row in enumerate(origa_a):
    add("ORIGA", row["image"], int(row["label"]), "val" if i in set(origa_val_idx) else "train", "combined")
for row in origa_b:
    add("ORIGA", row["image"], int(row["label"]), "test", "combined")

refuge = list(csv.DictReader(Path("data/audit/refuge_manifest.csv").open(encoding="utf-8")))
refuge_train = [row for row in refuge if row["split"] == "train"]
refuge_train_idx, refuge_val_idx = train_test_split(
    np.arange(len(refuge_train)),
    test_size=0.2,
    stratify=[row["label"] for row in refuge_train],
    random_state=SEED,
)
for i, row in enumerate(refuge_train):
    add("REFUGE", row["image"], int(row["label"]), "val" if i in set(refuge_val_idx) else "train", "combined")
for row in refuge:
    if row["split"] in ("val", "test"):
        add("REFUGE", row["image"], int(row["label"]), "test", "combined", source_split=row["split"])

drishti = list(csv.DictReader(Path("data/audit/drishti_gs1_kaggle_mirror_manifest.csv").open(encoding="utf-8")))
drishti_train = [row for row in drishti if row["official_style_split"] == "train"]
drishti_test = [row for row in drishti if row["official_style_split"] == "test"]
drishti_train_idx, drishti_val_idx = train_test_split(
    np.arange(len(drishti_train)),
    test_size=0.2,
    stratify=[row["label"] for row in drishti_train],
    random_state=SEED,
)
for i, row in enumerate(drishti_train):
    split = "val" if i in set(drishti_val_idx) else "train"
    add("DRISHTI-GS1", row["image"], row["label"] == "glaucoma", split, "drishti", row["label"])
for row in drishti_test:
    add("DRISHTI-GS1", row["image"], row["label"] == "glaucoma", "test", "drishti", row["label"])

with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)


class PooledDataset(Dataset):
    def __init__(self, indices, augment=False):
        self.indices = indices
        self.augment = augment
        self.archives = {
            "acrima": zipfile.ZipFile("data/raw/ACRIMA_Database.zip"),
            "rim": zipfile.ZipFile("data/raw/RIM-ONE-r3.zip"),
            "combined": zipfile.ZipFile("data/raw/ORIGA_candidate_figshare.zip"),
            "drishti": zipfile.ZipFile("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip"),
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        record = records[self.indices[i]]
        raw = self.archives[record["archive"]].read(record["member"])
        with Image.open(io.BytesIO(raw)).convert("RGB") as image:
            if record["archive"] == "rim":
                image = image.crop((0, 0, image.width // 2, image.height))
            image = ImageOps.pad(image, (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            pixels = np.asarray(image).copy()
        if self.augment and random.random() < 0.5:
            pixels = pixels[:, ::-1].copy()
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float() / 255
        return normalize(tensor), torch.tensor(record["label"], dtype=torch.long)

    def close(self):
        for archive in self.archives.values():
            archive.close()


weights = ResNet18_Weights.DEFAULT
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

train_indices = [i for i, row in enumerate(records) if row["split"] == "train"]
val_indices = [i for i, row in enumerate(records) if row["split"] == "val"]
test_indices = [i for i, row in enumerate(records) if row["split"] == "test"]

counts = Counter((row["dataset"], row["label"]) for i, row in enumerate(records) if i in train_indices)
dataset_class_weights = {key: len(train_indices) / (len(counts) * count) for key, count in counts.items()}
sample_weights = [dataset_class_weights[(records[i]["dataset"], records[i]["label"])] for i in train_indices]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_ds = PooledDataset(train_indices, augment=True)
val_ds = PooledDataset(val_indices)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = []
best_val_auc = -1.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for pixels, labels in train_loader:
        pixels, labels = pixels.to(device), labels.to(device)
        logits = model(pixels)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    probabilities = []
    truths = []
    datasets = []
    with torch.no_grad():
        for pixels, labels in val_loader:
            logits = model(pixels.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            truths.append(labels.numpy())
    probability = np.concatenate(probabilities)
    truth = np.concatenate(truths)
    auc = float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else float("nan")
    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": auc})
    if auc > best_val_auc:
        best_val_auc = auc
        torch.save({
            "model_state": model.state_dict(),
            "weights": "ResNet18_Weights.DEFAULT",
            "epoch": epoch,
            "seed": SEED,
        }, root / "best.pt")
    scheduler.step()
    print(history[-1], flush=True)

checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_ds = PooledDataset(test_indices)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
probability_groups = []
truth_groups = []
with torch.no_grad():
    for pixels, labels in test_loader:
        logits = model(pixels.to(device))
        probability_groups.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        truth_groups.append(labels.numpy())
probabilities = np.concatenate(probability_groups)
truths = np.concatenate(truth_groups)

test_rows = [records[i] for i in test_indices]
def score_positions(dataset_positions):
    dataset_rows = [test_rows[i] for i in dataset_positions]
    y = np.asarray([row["label"] for row in dataset_rows])
    p = probabilities[dataset_positions]
    predicted = (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "n_test": len(dataset_rows),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


metrics = {}
for dataset in sorted(set(row["dataset"] for row in test_rows)):
    metrics[dataset] = score_positions([i for i, row in enumerate(test_rows) if row["dataset"] == dataset])
refuge_by_source_split = {
    split: score_positions([i for i, row in enumerate(test_rows) if row["dataset"] == "REFUGE" and row["source_split"] == split])
    for split in ("val", "test")
}

summary = {
    "status": "five_source_mirror_finetuned_partial",
    "architecture": "fine-tuned ResNet-18",
    "device": str(device),
    "seed": SEED,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "protocol": "ACRIMA fixed split; RIM-ONE fixed split excluding suspects; ORIGA A/B split; REFUGE supplied train/val/test; DRISHTI Kaggle mirror 40/10/51",
    "history": history,
    "test_metrics": metrics,
    "refuge_test_metrics_by_source_split": refuge_by_source_split,
    "limitations": "DRISHTI-GS1 is a Kaggle mirror; ORIGA and REFUGE provenance caveats remain; final source policy not yet frozen.",
}
(root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

with (root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["dataset", "member", "source_split", "true_label", "probability_glaucoma", "predicted_label"])
    current = 0
    for row, probability, truth, predicted in zip(
        [records[i] for i in test_indices], probabilities, truths, (probabilities >= 0.5).astype(int)
    ):
        writer.writerow([row["dataset"], row["member"], row["source_split"], int(truth), float(probability), int(predicted)])

train_ds.close()
val_ds.close()
test_ds.close()
print(json.dumps(summary, indent=2), flush=True)
