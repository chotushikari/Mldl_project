"""Render predicted-class Grad-CAM examples for four held-out datasets."""
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.models import resnet18
from torchvision.transforms import Normalize, ToTensor


torch.set_num_threads(4)
root = Path("outputs/pooled_four_dataset_frozen")
out = root / "gradcam"
out.mkdir(exist_ok=True)
sources = {
    "ACRIMA": ("acrima_test_predictions.csv", "data/raw/ACRIMA_Database.zip"),
    "ORIGA": ("origa_test_predictions.csv", "data/raw/ORIGA_candidate_figshare.zip"),
    "REFUGE": ("refuge_test_predictions.csv", "data/raw/ORIGA_candidate_figshare.zip"),
    "RIM-ONE r3": ("rim_one_r3_test_predictions.csv", "data/raw/RIM-ONE-r3.zip"),
}
model = resnet18(weights=None)
model.fc = torch.nn.Linear(512, 1)
model.load_state_dict(torch.load(root / "resnet18_classifier.pt", map_location="cpu", weights_only=True))
model.eval()
to_tensor = ToTensor()
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
activations, gradients = [], []


def hook(_module, _inputs, output):
    activations.append(output)
    output.register_hook(lambda grad: gradients.append(grad))


model.layer4[-1].register_forward_hook(hook)
index = []
for dataset, (predictions_path, archive_path) in sources.items():
    rows = list(csv.DictReader((root / predictions_path).open(newline="", encoding="utf-8")))
    groups = {key: [] for key in ("TP", "TN", "FP", "FN")}
    for row in rows:
        y, p = int(row["true_label"]), int(row["predicted_label"])
        groups[("TP" if y else "TN") if y == p else ("FP" if p else "FN")].append(row)
    selected = [(key, group[0]) for key, group in groups.items() if group]
    folder = out / dataset.lower().replace(" ", "_").replace("-", "_")
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for outcome, row in selected:
            with Image.open(io.BytesIO(archive.read(row["member"]))) as source:
                image = source.convert("RGB")
                if dataset == "RIM-ONE r3":
                    image = image.crop((0, 0, image.width // 2, image.height))
                image = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
            tensor = normalize(to_tensor(image))[None]
            activations.clear()
            gradients.clear()
            model.zero_grad()
            logit = model(tensor)[0, 0]
            target = logit if int(row["predicted_label"]) else -logit
            target.backward()
            a = activations[-1][0].detach()
            g = gradients[-1][0].detach()
            cam = torch.relu((g.mean((1, 2))[:, None, None] * a).sum(0))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
            heat = Image.fromarray(np.uint8(cam.numpy() * 255)).resize((224, 224), Image.Resampling.BILINEAR)
            red = Image.new("RGB", (224, 224), (255, 30, 20))
            overlay = Image.blend(image, Image.composite(red, image, heat), 0.5)
            panel = Image.new("RGB", (448, 260), "white")
            panel.paste(image, (0, 0))
            panel.paste(overlay, (224, 0))
            label = f"{dataset} {outcome} | true={row['true_label']} pred={row['predicted_label']} p={float(row['probability_glaucoma']):.3f}"
            ImageDraw.Draw(panel).text((8, 233), label, fill="black")
            path = folder / (outcome + "_" + Path(row["member"]).stem + ".png")
            panel.save(path)
            index.append({"dataset": dataset, "outcome": outcome, "member": row["member"], "predicted_class_target": int(row["predicted_label"]), "panel": str(path).replace("\\", "/")})

(out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
print(f"Wrote {len(index)} predicted-class Grad-CAM panels across {len(sources)} datasets to {out}")
