"""Save held-out DRISHTI-GS1 mirror segmentation scores and visual panels."""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet_small import UNet


SIZE = 256
root = Path("outputs/drishti_gs1_mirror")
archive = zipfile.ZipFile("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip")
manifest = list(csv.DictReader(Path("data/audit/drishti_gs1_kaggle_mirror_manifest.csv").open(encoding="utf-8")))
split = list(csv.DictReader((root / "split.csv").open(encoding="utf-8")))
test_names = {row["image"] for row in split if row["split"] == "test"}
lookup = {row["image"]: row for row in manifest}
model = UNet()
model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
model.eval()

rows = []
with torch.no_grad():
    for name in sorted(test_names):
        record = lookup[name]
        with Image.open(io.BytesIO(archive.read(name))).convert("RGB") as image:
            image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        masks = []
        for key in ("disc_mask", "cup_mask"):
            with Image.open(io.BytesIO(archive.read(record[key]))).convert("L") as mask:
                mask = mask.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
                masks.append((np.asarray(mask) >= 128))
        pixels = np.asarray(image).copy()
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].float() / 255
        pred = model(tensor).sigmoid()[0].detach().numpy() > 0.5
        row = {"dataset": "DRISHTI-GS1", "source": "lokeshsaipureddi Kaggle mirror", "image": name}
        for i, region in enumerate(("disc", "cup")):
            p, y = pred[i], masks[i]
            inter = int(np.logical_and(p, y).sum())
            ps = int(p.sum())
            gs = int(y.sum())
            union = int(np.logical_or(p, y).sum())
            row[f"{region}_dice"] = (2 * inter + 1e-6) / (ps + gs + 1e-6)
            row[f"{region}_iou"] = (inter + 1e-6) / (union + 1e-6)
        rows.append(row)

with (root / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

out = root / "examples"
out.mkdir(parents=True, exist_ok=True)
index = []
ordered = sorted(rows, key=lambda r: (r["disc_dice"] + r["cup_dice"]) / 2)
for quality, record in (("worst", ordered[0]), ("best", ordered[-1])):
    row = lookup[record["image"]]
    with Image.open(io.BytesIO(archive.read(row["image"]))).convert("RGB") as image:
        image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    truth = []
    for key in ("disc_mask", "cup_mask"):
        with Image.open(io.BytesIO(archive.read(row[key]))).convert("L") as mask:
            mask = mask.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            truth.append((np.asarray(mask) >= 128))
    pixels = np.asarray(image).copy()
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].float() / 255
    pred = model(tensor).sigmoid()[0].detach().numpy() > 0.5
    panel = Image.new("RGB", (768, 256), "white")
    panel.paste(image, (0, 0))
    for panel_x, i in ((256, 0), (512, 1)):
        y = truth[i]
        p = pred[i]
        overlay = pixels.copy()
        overlap = y & p
        false_negative = y & ~p
        false_positive = p & ~y
        overlay[overlap] = 0.55 * overlay[overlap] + 0.45 * np.array([0, 220, 110])
        overlay[false_negative] = 0.55 * overlay[false_negative] + 0.45 * np.array([50, 120, 255])
        overlay[false_positive] = 0.55 * overlay[false_positive] + 0.45 * np.array([255, 170, 40])
        panel.paste(Image.fromarray(overlay.astype("uint8")), (panel_x, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 238), f"{quality} original", fill="black")
    draw.text((264, 238), "disc: green overlap / blue FN / orange FP", fill="black")
    draw.text((520, 238), "cup: same encoding", fill="black")
    path = out / f"{quality}_{Path(row['image']).stem}.png"
    panel.save(path)
    index.append({"quality": quality, "image": row["image"], "panel": str(path).replace("\\", "/"),
                  "disc_dice": record["disc_dice"], "cup_dice": record["cup_dice"]})

(out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
archive.close()
print(json.dumps({"n_test": len(rows), "panels": index}, indent=2))
