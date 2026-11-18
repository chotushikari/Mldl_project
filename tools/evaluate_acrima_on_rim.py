"""External test: ACRIMA-trained baseline on RIM-ONE r3 normal/glaucoma eyes."""
import csv
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18

torch.set_num_threads(4)
rows = [r for r in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8")) if r["label_original"] != "suspect"]
labels = np.array([int(r["label_original"] == "glaucoma") for r in rows])
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = torch.nn.Identity()
model.eval()
transform = weights.transforms()
feats = []
with zipfile.ZipFile("data/raw/RIM-ONE-r3.zip") as archive, torch.no_grad():
    for start in range(0, len(rows), 16):
        batch = []
        for row in rows[start:start+16]:
            with Image.open(io.BytesIO(archive.read(row["image"]))) as image:
                image = image.convert("RGB")
                image = image.crop((0, 0, image.width // 2, image.height))
                batch.append(transform(image))
        feats.append(model(torch.stack(batch)).numpy())
        print(f"features {min(start+16,len(rows))}/{len(rows)}", flush=True)
classifier = joblib.load("outputs/acrima_resnet18_frozen/classifier.joblib")
probs = classifier.predict_proba(np.concatenate(feats))[:, 1]
preds = (probs >= 0.5).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
metrics = {"train_dataset": "ACRIMA", "test_dataset": "RIM-ONE r3", "protocol": "left stereo view; suspects excluded; no RIM training or threshold tuning",
           "n_test": len(rows), "accuracy": accuracy_score(labels, preds), "precision": precision, "recall": recall,
           "f1": f1, "auc": roc_auc_score(labels, probs), "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
out = Path("outputs/acrima_resnet18_frozen")
(out / "rim_external_metrics.json").write_text(json.dumps(metrics, indent=2))
with (out / "rim_external_predictions.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "true_label", "probability_glaucoma", "predicted_label"])
    writer.writerows((row["image"], int(y), float(p), int(pred)) for row, y, p, pred in zip(rows, labels, probs, preds))
print(json.dumps(metrics, indent=2))
