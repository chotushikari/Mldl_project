"""Validate ACRIMA labels against the archive readme and make a visual audit sheet."""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

archive = Path("data/raw/ACRIMA_Database.zip")
out = Path("data/audit")
out.mkdir(parents=True, exist_ok=True)
rows = []
with zipfile.ZipFile(archive) as z:
    names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
    for name in names:
        filename = Path(name).name
        match = re.fullmatch(r"Im(\d{3})(_g)?_ACRIMA\.jpg", filename, re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected ACRIMA filename: {filename}")
        with Image.open(io.BytesIO(z.read(name))) as img:
            img.verify()
        rows.append({"dataset": "ACRIMA", "archive": str(archive), "member": name, "image_id": match.group(1), "label": "glaucoma" if match.group(2) else "normal", "has_disc_mask": False, "has_cup_mask": False})
    n_g = sum(r["label"] == "glaucoma" for r in rows)
    assert (len(rows), n_g) == (705, 396), (len(rows), n_g)
    chosen = [next(r for r in rows if r["label"] == label) for label in ("normal", "glaucoma")]
    sheet = Image.new("RGB", (800, 430), "#f6f8f9")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(chosen):
        with Image.open(io.BytesIO(z.read(row["member"]))) as img:
            img.thumbnail((370, 350))
            sheet.paste(img, (20 + idx * 400 + (370 - img.width)//2, 35))
        draw.text((20 + idx * 400, 395), f"{row['label']} | {Path(row['member']).name}", fill="black")
    sheet.save(out / "acrima_samples.jpg", quality=92)
with (out / "acrima_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"ACRIMA verified: {len(rows)} images, {n_g} glaucoma, {len(rows)-n_g} normal; no masks")
