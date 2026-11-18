"""Reproducible ACRIMA-only frozen ResNet-18 feature baseline (CPU compatible)."""
from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import joblib
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights, resnet18

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)
manifest = list(csv.DictReader(Path("data/audit/acrima_manifest.csv").open(encoding="utf-8")))
labels = np.array([int(r["label"] == "glaucoma") for r in manifest])
indices = np.arange(len(manifest))
trainval, test = train_test_split(indices, test_size=0.20, stratify=labels, random_state=SEED)
train, val = train_test_split(trainval, test_size=0.1875, stratify=labels[trainval], random_state=SEED)
out = Path("outputs/acrima_resnet18_frozen")
out.mkdir(parents=True, exist_ok=True)
split = {int(i): name for name, arr in (("train", train), ("val", val), ("test", test)) for i in arr}
with (out / "split.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "member", "label", "split"])
    writer.writerows((r["image_id"], r["member"], r["label"], split[i]) for i, r in enumerate(manifest))

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
transform = weights.transforms()
features = []
with zipfile.ZipFile("data/raw/ACRIMA_Database.zip") as archive, torch.no_grad():
    for start in range(0, len(manifest), 16):
        batch = []
        for row in manifest[start:start + 16]:
            with Image.open(io.BytesIO(archive.read(row["member"]))) as image:
                batch.append(transform(image.convert("RGB")))
        features.append(model(torch.stack(batch)).numpy())
        print(f"features {min(start+16,len(manifest))}/{len(manifest)}", flush=True)
X = np.concatenate(features)
candidate_cs = [0.01, 0.1, 1.0, 10.0]
best = None
for c in candidate_cs:
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=SEED).fit(X[train], labels[train])
    auc = roc_auc_score(labels[val], clf.predict_proba(X[val])[:, 1])
    if best is None or auc > best[0]:
        best = (auc, c)
clf = LogisticRegression(C=best[1], class_weight="balanced", max_iter=1000, random_state=SEED).fit(X[trainval], labels[trainval])
joblib.dump(clf, out / "classifier.joblib")
p = clf.predict_proba(X[test])[:, 1]
yhat = (p >= 0.5).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(labels[test], yhat, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(labels[test], yhat, labels=[0, 1]).ravel()
metrics = {"dataset": "ACRIMA", "experiment": "frozen ResNet-18 features + logistic regression", "seed": SEED,
           "n_train": len(train), "n_val": len(val), "n_test": len(test), "selection_val_auc": best[0], "selected_C": best[1],
           "test_accuracy": accuracy_score(labels[test], yhat), "test_precision": precision, "test_recall": recall,
           "test_f1": f1, "test_auc": roc_auc_score(labels[test], p), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "true_label", "probability_glaucoma", "predicted_label"])
    writer.writerows((manifest[i]["image_id"], labels[i], float(prob), int(pred)) for i, prob, pred in zip(test, p, yhat))
print(json.dumps(metrics, indent=2), flush=True)
