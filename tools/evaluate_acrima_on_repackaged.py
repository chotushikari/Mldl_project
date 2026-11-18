"""Untuned external classification on ORIGA and labeled REFUGE validation images."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torchvision.models import ResNet18_Weights, resnet18


torch.set_num_threads(4)
base = Path("outputs/acrima_resnet18_frozen")
archive_path = Path("data/raw/ORIGA_candidate_figshare.zip")
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 1)
model.load_state_dict(torch.load(base / "resnet18_classifier.pt", map_location="cpu", weights_only=True))
model.eval()
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()


def evaluate(name, rows, output_stem):
    result_rows = []
    with zipfile.ZipFile(archive_path) as archive, torch.no_grad():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            images = []
            for record in batch:
                with Image.open(io.BytesIO(archive.read(record["image"]))) as source:
                    images.append(transform(source.convert("RGB")))
            probabilities = model(torch.stack(images)).sigmoid().flatten().numpy()
            for record, probability in zip(batch, probabilities):
                result_rows.append({
                    "image": record["image"], "true_label": int(record["label"]),
                    "probability_glaucoma": float(probability),
                    "predicted_label": int(probability >= 0.5),
                })
            print(f"{name}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    y = np.array([r["true_label"] for r in result_rows])
    p = np.array([r["probability_glaucoma"] for r in result_rows])
    predicted = np.array([r["predicted_label"] for r in result_rows])
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics = {
        "train_dataset": "ACRIMA", "test_dataset": name,
        "protocol": "frozen ACRIMA checkpoint; ImageNet ResNet-18 center-crop preprocessing; no target training or threshold tuning",
        "n_test": len(rows), "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auc": float(roc_auc_score(y, p)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    (base / f"{output_stem}_external_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (base / f"{output_stem}_external_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(metrics, indent=2), flush=True)


origa = list(csv.DictReader(Path("data/audit/origa_manifest.csv").open(newline="", encoding="utf-8")))
refuge = [r for r in csv.DictReader(Path("data/audit/refuge_manifest.csv").open(newline="", encoding="utf-8")) if r["split"] == "val"]
assert len(origa) == 650 and len(refuge) == 400
evaluate("ORIGA", origa, "origa")
evaluate("REFUGE validation", refuge, "refuge_val")
