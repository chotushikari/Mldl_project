"""Evaluate the frozen four-dataset model on original-source REFUGE test labels."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision.models import resnet18
from torchvision.transforms import Normalize, ToTensor


torch.set_num_threads(4)
root = Path("outputs/pooled_four_dataset_frozen")
records = [r for r in csv.DictReader(Path("data/audit/refuge_manifest.csv").open(newline="", encoding="utf-8")) if r["split"] == "test"]
assert len(records) == 400 and all(r["label"] in ("0", "1") for r in records)
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 1)
model.load_state_dict(torch.load(root / "resnet18_classifier.pt", map_location="cpu", weights_only=True))
model.eval()
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
outputs = []
with zipfile.ZipFile("data/raw/ORIGA_candidate_figshare.zip") as archive, torch.no_grad():
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        images = []
        for record in batch:
            with Image.open(io.BytesIO(archive.read(record["image"]))) as source:
                image = ImageOps.pad(source.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
                images.append(normalize(to_tensor(image)))
        probability = model(torch.stack(images)).sigmoid().flatten().numpy()
        for record, p in zip(batch, probability):
            outputs.append({"image": record["image"], "true_label": int(record["label"]), "probability_glaucoma": float(p), "predicted_label": int(p >= 0.5)})
        print(f"REFUGE official test: {min(start + len(batch), len(records))}/{len(records)}", flush=True)

y = np.array([r["true_label"] for r in outputs])
p = np.array([r["probability_glaucoma"] for r in outputs])
predicted = np.array([r["predicted_label"] for r in outputs])
precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
metrics = {
    "train_protocol": "pooled four-dataset frozen ResNet-18 baseline; no REFUGE validation/test tuning",
    "test_dataset": "REFUGE official test 400",
    "label_source": "original-source REFUGE-Test-GT/Glaucoma_label_and_Fovea_location.xlsx",
    "n_test": len(outputs), "accuracy": float(accuracy_score(y, predicted)),
    "precision": float(precision), "recall": float(recall), "f1": float(f1),
    "auc": float(roc_auc_score(y, p)),
    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
}
(root / "refuge_official_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
with (root / "refuge_official_test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(outputs[0]))
    writer.writeheader()
    writer.writerows(outputs)
print(json.dumps(metrics, indent=2))
