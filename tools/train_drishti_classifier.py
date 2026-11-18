"""Frozen ResNet-18 classifier baseline on the Kaggle DRISHTI-GS1 mirror."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Normalize, ToTensor


SEED = 42
root = Path("outputs/drishti_gs1_mirror_classifier")
root.mkdir(parents=True, exist_ok=True)
archive = zipfile.ZipFile("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip")
rows = list(csv.DictReader(Path("data/audit/drishti_gs1_kaggle_mirror_manifest.csv").open(encoding="utf-8")))
train_pool = [row for row in rows if row["official_style_split"] == "train"]
test_rows = [row for row in rows if row["official_style_split"] == "test"]
train_idx, val_idx = train_test_split(
    np.arange(len(train_pool)),
    test_size=0.2,
    stratify=[row["label"] for row in train_pool],
    random_state=SEED,
)
train_rows = [train_pool[i] for i in train_idx]
val_rows = [train_pool[i] for i in val_idx]

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model.eval()
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def features(records):
    values = []
    with torch.no_grad():
        for start in range(0, len(records), 8):
            batch = []
            for record in records[start:start + 8]:
                with Image.open(io.BytesIO(archive.read(record["image"]))).convert("RGB") as image:
                    image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
                    batch.append(normalize(to_tensor(image)))
            values.append(model(torch.stack(batch)).numpy())
    return np.concatenate(values)


X_train = features(train_rows)
X_val = features(val_rows)
X_trainval = features(train_rows + val_rows)
X_test = features(test_rows)
y_train = np.asarray([row["label"] == "glaucoma" for row in train_rows], dtype=int)
y_val = np.asarray([row["label"] == "glaucoma" for row in val_rows], dtype=int)
y_trainval = np.asarray([row["label"] == "glaucoma" for row in train_rows + val_rows], dtype=int)
y_test = np.asarray([row["label"] == "glaucoma" for row in test_rows], dtype=int)

trials = []
for c in (0.01, 0.1, 1.0, 10.0):
    classifier = LogisticRegression(C=c, max_iter=1000, random_state=SEED)
    classifier.fit(X_train, y_train)
    probability = classifier.predict_proba(X_val)[:, 1]
    trials.append({"C": c, "val_auc": float(roc_auc_score(y_val, probability))})
selected = max(trials, key=lambda item: item["val_auc"])["C"]
classifier = LogisticRegression(C=selected, max_iter=1000, random_state=SEED)
classifier.fit(X_trainval, y_trainval)
joblib.dump(classifier, root / "classifier.joblib")

probability = classifier.predict_proba(X_test)[:, 1]
predicted = (probability >= 0.5).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
metrics = {
    "status": "drishti_gs1_kaggle_mirror_partial_classification",
    "dataset": "DRISHTI-GS1",
    "source": "lokeshsaipureddi Kaggle mirror",
    "architecture": "frozen ImageNet ResNet-18 features + logistic head",
    "seed": SEED,
    "n_train": len(train_rows),
    "n_val": len(val_rows),
    "n_test": len(test_rows),
    "accuracy": float(accuracy_score(y_test, predicted)),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "auc": float(roc_auc_score(y_test, probability)),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "selected_C": selected,
    "val_trials": trials,
}
with (root / "metrics.json").open("w", encoding="utf-8") as stream:
    json.dump(metrics, stream, indent=2)
with (root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["dataset", "member", "true_label", "probability_glaucoma", "predicted_label"])
    writer.writerows([
        ("DRISHTI-GS1", row["image"], int(y), float(p), int(yhat))
        for row, y, p, yhat in zip(test_rows, y_test, probability, predicted)
    ])
split = [{"image": row["image"], "label": row["label"], "split": "train"} for row in train_rows]
split += [{"image": row["image"], "label": row["label"], "split": "val"} for row in val_rows]
split += [{"image": row["image"], "label": row["label"], "split": "test"} for row in test_rows]
with (root / "split.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["image", "label", "split"])
    writer.writeheader()
    writer.writerows(split)

print(json.dumps(metrics, indent=2))
archive.close()
