"""Audit third-party Kaggle DRISHTI-GS mirror structure without final-use claims."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


archive = Path("data/provisional/drishti_gs1/drishti_gs1_kaggle_mirror.zip")
audit = Path("data/provisional/drishti_gs1/audit")
audit.mkdir(parents=True, exist_ok=True)

if not archive.exists():
    raise SystemExit("Mirror archive is not present; run tools/download_drishti_kaggle_inventory.py first")

image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
with zipfile.ZipFile(archive) as z:
    members = z.namelist()
    image_members = sorted(name for name in members if Path(name).suffix.lower() in image_suffixes)
    actual_images = sorted(
        name for name in image_members
        if "/images/" in name.lower()
        and "/gt/" not in name.lower()
        and "mask" not in name.lower()
        and "ground" not in name.lower()
    )
    actual_masks = sorted(
        name for name in image_members
        if "/gt/" in name.lower()
        or "/test_gt/" in name.lower()
        or "mask" in name.lower()
        or "ground" in name.lower()
    )
    sample_sheet = [name for name in image_members[:12]]
    suffix_counts = Counter(Path(name).suffix.lower() for name in image_members)
    label_counts = Counter(
        "glaucoma" if "/glaucoma/" in name.lower()
        else "normal" if "/normal/" in name.lower()
        else "unknown"
        for name in actual_images
    )
    od_soft_masks = sorted(
        name for name in actual_masks
        if "odsegsoftmap" in name.lower()
    )
    cup_soft_masks = sorted(
        name for name in actual_masks
        if "cupsegsoftmap" in name.lower()
    )

    rows = []
    for name in actual_images:
        lowered = name.lower()
        split = "train" if "/train/" in lowered or "train" in Path(name).parts[0].lower() else "test" if "/test/" in lowered or "test" in Path(name).parts[0].lower() else "unclassified"
        rows.append({
            "member": name,
            "split": split,
            "suffix": Path(name).suffix.lower(),
        })

    manifest = audit / "drishti_gs1_kaggle_mirror_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["member", "split", "suffix"])
        writer.writeheader()
        writer.writerows(rows)

    sample = [name for name in image_members if "/images/" in name.lower()][:8]
    if sample:
        canvas = Image.new("RGB", (1000, 560), "white")
        draw = ImageDraw.Draw(canvas)
        for i, name in enumerate(sample):
            with Image.open(io.BytesIO(z.read(name))) as image:
                image = image.convert("RGB")
                image.thumbnail((220, 220))
            x = 20 + (i % 4) * 245
            y = 20 + (i // 4) * 260
            canvas.paste(image, (x + (220 - image.width) // 2, y))
            draw.text((x, y + 225), Path(name).name, fill="black")
        canvas.save(audit / "drishti_gs1_kaggle_mirror_samples.jpg", quality=92)

summary = {
    "status": "third_party_mirror_inventory_only",
    "archive": str(archive),
    "member_count": len(members),
    "all_png_count": len(image_members),
    "image_count": len(actual_images),
    "mask_count": len(actual_masks),
    "od_soft_mask_count": len(od_soft_masks),
    "cup_soft_mask_count": len(cup_soft_masks),
    "provisional_label_counts": dict(label_counts),
    "image_suffix_counts": dict(suffix_counts),
    "split_counts": dict(Counter(row["split"] for row in rows)),
    "sample_members": sample_sheet,
    "final_training_eligible": False,
    "official_verification_required": [
        "official archive contents",
        "101 image count",
        "50/51 official split",
        "expert mask directories",
        "soft masks and cup-to-disc ratio availability",
        "authorized test-ground-truth provenance",
    ],
}
(audit / "drishti_gs1_kaggle_mirror_inventory.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
