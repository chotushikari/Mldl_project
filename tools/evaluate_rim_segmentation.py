"""Save held-out per-image optic disc/cup metrics from the saved RIM model."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet


torch.set_num_threads(4)
root = Path("outputs/rim_unet_baseline")
manifest = {r["image"]: r for r in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))}
test_names = [r["image"] for r in csv.DictReader((root / "split.csv").open(encoding="utf-8")) if r["split"] == "test"]
model = UNet()
model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
model.eval()


def load_left(z, name, is_mask=False):
    image = Image.open(io.BytesIO(z.read(name))).convert("L" if is_mask else "RGB")
    return image.crop((0, 0, image.width // 2, image.height)).resize(
        (256, 256), Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    )


rows = []
with zipfile.ZipFile("data/raw/RIM-ONE-r3.zip") as archive, torch.no_grad():
    for name in test_names:
        record = manifest[name]
        image = np.asarray(load_left(archive, name)).copy()
        truth = [np.asarray(load_left(archive, record[key], True)) > 127 for key in ("disc_mask", "cup_mask")]
        x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()[None] / 255
        pred = model(x).sigmoid()[0].numpy() > 0.5
        result = {"image": name, "label_original": record["label_original"]}
        for i, region in enumerate(("disc", "cup")):
            p, y = pred[i], truth[i]
            intersection = int(np.logical_and(p, y).sum())
            predicted = int(p.sum())
            actual = int(y.sum())
            union = int(np.logical_or(p, y).sum())
            result[f"{region}_dice"] = (2 * intersection + 1e-6) / (predicted + actual + 1e-6)
            result[f"{region}_iou"] = (intersection + 1e-6) / (union + 1e-6)
        rows.append(result)

saved = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
assert len(rows) == saved["n_test"]
for region in ("disc", "cup"):
    for metric in ("dice", "iou"):
        key = f"{region}_{metric}"
        assert abs(float(np.mean([r[key] for r in rows])) - saved[f"test_{key}"]) < 1e-6, key

destination = root / "test_predictions.csv"
with destination.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Verified and wrote {len(rows)} held-out per-image scores to {destination}")
