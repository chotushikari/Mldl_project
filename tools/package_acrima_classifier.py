"""Package the measured frozen-feature classifier as one PyTorch checkpoint."""
import csv
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


torch.set_num_threads(4)
root = Path("outputs/acrima_resnet18_frozen")
clf = joblib.load(root / "classifier.joblib")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(512, 1)
with torch.no_grad():
    model.fc.weight.copy_(torch.from_numpy(clf.coef_).float())
    model.fc.bias.copy_(torch.from_numpy(clf.intercept_).float())
model.eval()
transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
manifest = {r["image_id"]: r for r in csv.DictReader(Path("data/audit/acrima_manifest.csv").open(encoding="utf-8"))}
predictions = list(csv.DictReader((root / "test_predictions.csv").open(encoding="utf-8")))

# Float32 conversion changes a logistic probability only by numerical rounding.
with zipfile.ZipFile("data/raw/ACRIMA_Database.zip") as archive, torch.no_grad():
    for record in predictions[:5]:
        image = Image.open(io.BytesIO(archive.read(manifest[record["image_id"]]["member"]))).convert("RGB")
        probability = float(model(transform(image)[None]).sigmoid().item())
        expected = float(record["probability_glaucoma"])
        assert abs(probability - expected) < 1e-5, (record["image_id"], probability, expected)

checkpoint = root / "resnet18_classifier.pt"
torch.save(model.state_dict(), checkpoint)
metadata = {
    "architecture": "torchvision.models.resnet18 with one-logit linear head",
    "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
    "positive_class": "glaucoma",
    "probability": "sigmoid(logit)",
    "threshold": 0.5,
    "source_experiment": "outputs/acrima_resnet18_frozen/metrics.json",
    "checkpoint_bytes": checkpoint.stat().st_size,
    "verification": "First five saved ACRIMA test probabilities agree within 1e-5",
}
(root / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(checkpoint, checkpoint.stat().st_size, "bytes; five probabilities verified")
