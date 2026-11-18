"""Check exact archive-image duplicates across measured train/val/test splits."""
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path


experiments = (
    ("ACRIMA", Path("data/raw/ACRIMA_Database.zip"), Path("outputs/acrima_resnet18_frozen/split.csv"), "member"),
    ("RIM-ONE r3", Path("data/raw/RIM-ONE-r3.zip"), Path("outputs/rim_unet_baseline/split.csv"), "image"),
)
report = {}
for dataset, archive_path, split_path, member_key in experiments:
    rows = list(csv.DictReader(split_path.open(newline="", encoding="utf-8")))
    groups = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for row in rows:
            digest = hashlib.sha256(archive.read(row[member_key])).hexdigest()
            groups[digest].append({"member": row[member_key], "split": row["split"]})
    duplicates = [members for members in groups.values() if len(members) > 1]
    cross_split = [members for members in duplicates if len({m["split"] for m in members}) > 1]
    report[dataset] = {"n_images": len(rows), "exact_duplicate_groups": duplicates, "cross_split_duplicate_groups": cross_split}
    print(f"{dataset}: {len(rows)} images; {len(duplicates)} exact duplicate groups; {len(cross_split)} cross-split groups")
    assert not cross_split, f"Exact-image leakage in {dataset}"

destination = Path("outputs/split_duplicate_audit.json")
destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Wrote", destination)
