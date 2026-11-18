"""Audit original RIM-ONE r3 archive labels and paired average masks."""
import collections
import csv
import zipfile
from pathlib import Path

archive = Path("data/raw/RIM-ONE-r3.zip")
out = Path("data/audit")
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as z:
    names = set(z.namelist())
    images = sorted(n for n in names if n.lower().endswith(".jpg"))
    rows = []
    for image in images:
        prefix = Path(image).name.split("-")[0]
        assert prefix in {"N", "G", "S"}, image
        base = image.rsplit("/Stereo Images/", 1)[0]
        stem = Path(image).stem
        disc = f"{base}/Average_masks/{stem}-Disc-Avg.png"
        cup = f"{base}/Average_masks/{stem}-Cup-Avg.png"
        assert disc in names and cup in names, (image, disc, cup)
        rows.append({"image": image, "label_original": {"N":"normal","G":"glaucoma","S":"suspect"}[prefix], "disc_mask": disc, "cup_mask": cup})
assert collections.Counter(r["label_original"] for r in rows) == {"normal":85,"glaucoma":39,"suspect":35}
with (out / "rim_one_r3_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("RIM-ONE r3 verified: 85 normal, 39 glaucoma, 35 suspect; all 159 have disc and cup average masks")
