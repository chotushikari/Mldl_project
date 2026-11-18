"""Matched whole-image vs predicted-disc-crop ResNet-18 comparison on RIM-ONE r3."""
from __future__ import annotations

import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Normalize, ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

root = Path("outputs/rim_whole_vs_crop")
root.mkdir(parents=True, exist_ok=True)
manifest = {
    row["image"]: row
    for row in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))
}
seg_split = list(csv.DictReader(Path("outputs/rim_unet_baseline/split.csv").open(encoding="utf-8")))
binary = [
    row for row in seg_split
    if manifest[row["image"]]["label_original"] in ("normal", "glaucoma")
]

segmenter = UNet()
segmenter.load_state_dict(torch.load(
    "outputs/rim_unet_baseline/best.pt", map_location="cpu", weights_only=True
))
segmenter.eval()

feature_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
feature_model.fc = torch.nn.Identity()
feature_model.eval()
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def left_image(archive: zipfile.ZipFile, record: dict) -> Image.Image:
    with Image.open(io.BytesIO(archive.read(record["image"]))).convert("RGB") as image:
        left = image.crop((0, 0, image.width // 2, image.height)).resize((256, 256), Image.Resampling.BICUBIC)
    return left


def predicted_disc(image: Image.Image) -> np.ndarray:
    tensor = torch.from_numpy(
        np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
    )[None].float() / 255
    with torch.no_grad():
        logits = segmenter(tensor)
    return logits.sigmoid()[0, 0].numpy() > 0.5


def crop_disc(image: Image.Image, disc: np.ndarray) -> Image.Image:
    ys, xs = np.where(disc)
    if len(xs) == 0:
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    size = int(max(x1 - x0, y1 - y0) * 1.5)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    box = (
        max(0, cx - size // 2),
        max(0, cy - size // 2),
        min(image.width, cx + (size + 1) // 2),
        min(image.height, cy + (size + 1) // 2),
    )
    cropped = image.crop(box)
    return ImageOps.pad(cropped, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def whole_image(image: Image.Image) -> Image.Image:
    return ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


X = {"whole": [], "crop": []}
labels = []
splits = []
members = []

with zipfile.ZipFile("data/raw/RIM-ONE-r3.zip") as archive, torch.no_grad():
    for row in binary:
        record = manifest[row["image"]]
        image = left_image(archive, record)
        disc = predicted_disc(image)
        variants = {
            "whole": whole_image(image),
            "crop": crop_disc(image, disc),
        }
        for name, prepared in variants.items():
            tensor = normalize(to_tensor(prepared))[None]
            feature = feature_model(tensor).numpy()[0]
            X[name].append(feature)
        labels.append(record["label_original"] == "glaucoma")
        splits.append(row["split"])
        members.append(row["image"])

X = {name: np.asarray(values) for name, values in X.items()}
labels = np.asarray(labels, dtype=int)
splits = np.asarray(splits)
train = np.where(splits == "train")[0]
val = np.where(splits == "val")[0]
trainval = np.where(splits != "test")[0]
test = np.where(splits == "test")[0]

comparison = {"dataset": "RIM-ONE r3", "seed": SEED, "n_train": len(train),
              "n_val": len(val), "n_test": len(test), "protocol": {
                  "whole": "left stereo fundus view letterboxed to 224x224",
                  "crop": "disc crop from saved U-Net predicted disc mask in every split; no classification ground-truth mask crop",
              }, "selection_metric": "validation AUC", "candidates": (0.01, 0.1, 1.0, 10.0)}

for name in ("whole", "crop"):
    trials = []
    for c in comparison["candidates"]:
        classifier = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
        classifier.fit(X[name][train], labels[train])
        probabilities = classifier.predict_proba(X[name][val])[:, 1]
        trials.append({"C": c, "val_auc": float(roc_auc_score(labels[val], probabilities))})
    best_c = max(trials, key=lambda item: item["val_auc"])["C"]
    classifier = LogisticRegression(C=best_c, max_iter=1000, random_state=SEED)
    classifier.fit(X[name][trainval], labels[trainval])
    joblib.dump(classifier, root / f"{name}_classifier.joblib")

    probabilities = classifier.predict_proba(X[name][test])[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels[test], predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(labels[test], predicted, labels=[0, 1]).ravel()
    metrics = {
        "n_test": len(test),
        "accuracy": float(accuracy_score(labels[test], predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(labels[test], probabilities)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "selected_C": best_c,
        "val_trials": trials,
    }
    comparison[name] = metrics
    with (root / f"{name}_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "member", "true_label", "probability_glaucoma", "predicted_label"])
        writer.writerows([
            ("RIM-ONE r3", members[i], int(labels[i]), float(probability), int(yhat))
            for i, probability, yhat in zip(test, probabilities, predicted)
        ])

with (root / "metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(comparison, stream, indent=2)
with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["member", "label", "split"])
    writer.writeheader()
    writer.writerows({"member": members[i], "label": int(labels[i]), "split": splits[i]} for i in range(len(members)))

print(json.dumps(comparison, indent=2))
