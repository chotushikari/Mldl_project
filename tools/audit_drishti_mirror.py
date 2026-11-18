"""Create the working DRISHTI-GS1 manifest from the requested Kaggle mirror."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


archive = Path("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip")
audit = Path("data/audit")
audit.mkdir(parents=True, exist_ok=True)
rows = []

with zipfile.ZipFile(archive) as z:
    members = set(z.namelist())
    for name in sorted(n for n in members if n.lower().endswith(".png") and "/images/" in n.lower()):
        lowered = name.lower()
        split = "train" if "/training/" in lowered else "test" if "/test/" in lowered else None
        label = "glaucoma" if "/glaucoma/" in lowered else "normal"
        stem = Path(name).stem
        if split == "train":
            disc = f"Training-20211018T055246Z-001/Training/GT/{stem}/SoftMap/{stem}_ODsegSoftmap.png"
            cup = f"Training-20211018T055246Z-001/Training/GT/{stem}/SoftMap/{stem}_cupsegSoftmap.png"
        else:
            disc = f"Test-20211018T060000Z-001/Test/Test_GT/{stem}/SoftMap/{stem}_ODsegSoftmap.png"
            cup = f"Test-20211018T060000Z-001/Test/Test_GT/{stem}/SoftMap/{stem}_cupsegSoftmap.png"
        if disc not in members or cup not in members:
            raise AssertionError((name, disc, cup))
        with Image.open(io.BytesIO(z.read(name))) as image:
            image.verify()
        rows.append({
            "dataset": "DRISHTI-GS1",
            "source": "lokeshsaipureddi Kaggle mirror",
            "image": name,
            "disc_mask": disc,
            "cup_mask": cup,
            "label": label,
            "official_style_split": split,
        })

assert len(rows) == 101
assert Counter(r["official_style_split"] for r in rows) == {"train": 50, "test": 51}
assert Counter(r["label"] for r in rows) == {"glaucoma": 70, "normal": 31}

manifest = audit / "drishti_gs1_kaggle_mirror_manifest.csv"
with manifest.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

chosen = [next(r for r in rows if r["label"] == label) for label in ("normal", "glaucoma")]
with zipfile.ZipFile(archive) as z:
    canvas = Image.new("RGB", (800, 430), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(chosen):
        with Image.open(io.BytesIO(z.read(row["image"]))) as image:
            image = image.convert("RGB")
            image.thumbnail((370, 350))
            canvas.paste(image, (20 + i * 400 + (370 - image.width) // 2, 35))
        draw.text((20 + i * 400, 395), f"{row['label']} | {Path(row['image']).name}", fill="black")
    canvas.save(audit / "drishti_gs1_kaggle_mirror_samples.jpg", quality=92)

summary = {
    "dataset": "DRISHTI-GS1",
    "source": "Kaggle mirror lokeshsaipureddi",
    "images": len(rows),
    "labels": dict(Counter(r["label"] for r in rows)),
    "official_style_split": dict(Counter(r["official_style_split"] for r in rows)),
    "disc_masks": len(rows),
    "cup_masks": len(rows),
    "manifest": str(manifest),
    "provisional": True,
}
(audit / "drishti_gs1_kaggle_mirror_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
