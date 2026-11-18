"""Render pooled segmentation panels and save per-image held-out scores."""
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
root = Path("outputs/pooled_segmentation")
out = root / "examples"
out.mkdir(parents=True, exist_ok=True)
split = list(csv.DictReader((root / "split.csv").open(encoding="utf-8")))
rim_manifest = {
    row["image"]: row
    for row in csv.DictReader(Path("data/audit/rim_one_r3_manifest.csv").open(encoding="utf-8"))
}
model = UNet()
model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
model.eval()


def load_record(row):
    if row["dataset"] == "RIM-ONE r3":
        archive = rim
        rim_row = rim_manifest[row["image"]]
        with Image.open(io.BytesIO(archive.read(row["image"]))).convert("RGB") as image:
            image = image.crop((0, 0, image.width // 2, image.height)).resize(
                (SIZE, SIZE), Image.Resampling.BICUBIC
            )
        masks = []
        for key in ("disc_mask", "cup_mask"):
            with Image.open(io.BytesIO(archive.read(rim_row[key]))).convert("L") as mask:
                mask = mask.crop((0, 0, mask.width // 2, mask.height)).resize(
                    (SIZE, SIZE), Image.Resampling.NEAREST
                )
                masks.append(np.asarray(mask) > 127)
        truth = np.stack(masks).astype("float32")
    else:
        with Image.open(io.BytesIO(combined.read(row["image"]))).convert("RGB") as image:
            image = image.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        with Image.open(io.BytesIO(combined.read(row["mask"]))).convert("L") as mask:
            mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
            values = np.asarray(mask)
        truth = np.stack((values == 1, values == 2)).astype("float32")
    return image, truth


with zipfile.ZipFile("data/raw/RIM-ONE-r3.zip") as rim, zipfile.ZipFile("data/raw/ORIGA_candidate_figshare.zip") as combined, torch.no_grad():
    rows = []
    for row in split:
        if row["split"] != "test":
            continue
        image, truth = load_record(row)
        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
        )[None].float() / 255
        pred = model(tensor).sigmoid()[0].numpy() > 0.5
        result = {
            "dataset": row["dataset"],
            "image": row["image"],
        }
        for i, region in enumerate(("disc", "cup")):
            p = pred[i]
            y = truth[i] > 0
            intersection = int(np.logical_and(p, y).sum())
            predicted = int(p.sum())
            actual = int(y.sum())
            union = int(np.logical_or(p, y).sum())
            result[f"{region}_dice"] = (2 * intersection + 1e-6) / (predicted + actual + 1e-6)
            result[f"{region}_iou"] = (intersection + 1e-6) / (union + 1e-6)
        rows.append(result)

    destination = root / "test_predictions.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    index = []
    for dataset in ("RIM-ONE r3", "ORIGA", "REFUGE"):
        subset = [row for row in rows if row["dataset"] == dataset]
        ordered = sorted(subset, key=lambda r: (r["disc_dice"] + r["cup_dice"]) / 2)
        selected = [("worst", ordered[0]), ("best", ordered[-1])]
        for quality, result in selected:
            row = next(r for r in split if r["dataset"] == dataset and r["image"] == result["image"])
            image, truth = load_record(row)
            tensor = torch.from_numpy(
                np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
            )[None].float() / 255
            pred = model(tensor).sigmoid()[0].numpy() > 0.5
            panel = Image.new("RGB", (768, 256), "white")
            panel.paste(image, (0, 0))
            for panel_x, region_index in ((256, 0), (512, 1)):
                region = truth[region_index] > 0
                predicted = pred[region_index]
                overlay = np.asarray(image).copy()
                overlay[region & predicted] = 0.55 * overlay[region & predicted] + 0.45 * np.array([0, 220, 110])
                overlay[region & ~predicted] = 0.55 * overlay[region & ~predicted] + 0.45 * np.array([50, 120, 255])
                overlay[predicted & ~region] = 0.55 * overlay[predicted & ~region] + 0.45 * np.array([255, 170, 40])
                panel.paste(Image.fromarray(overlay.astype("uint8")), (panel_x, 0))
            draw = ImageDraw.Draw(panel)
            draw.text((8, 238), f"{dataset} {quality} original", fill="black")
            draw.text((264, 238), "green overlap / blue FN / orange FP: disc", fill="black")
            draw.text((520, 238), "same encoding: cup", fill="black")
            path = out / f"{quality}_{dataset.lower().replace(' ', '_').replace('-', '_')}_{Path(result['image']).stem}.png"
            panel.save(path)
            index.append({
                "dataset": dataset,
                "quality": quality,
                "image": result["image"],
                "panel": str(path).replace("\\", "/"),
                "disc_dice": result["disc_dice"],
                "cup_dice": result["cup_dice"],
            })

(out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} per-image scores and {len(index)} pooled segmentation panels")
