"""Recover official REFUGE test labels and check the repackaged archive against RAR samples."""
import csv
import io
import json
import shutil
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
from PIL import Image


rar = Path("data/raw/REFUGE_figshare.rar")
combined = Path("data/raw/ORIGA_candidate_figshare.zip")
audit = Path("data/audit")
if shutil.which("7z"):
    listed = subprocess.run(["7z", "l", "-slt", str(rar)], capture_output=True, text=True, check=True).stdout
    members = set()
    for line in listed.splitlines():
        if line.startswith("Path = ") and line != "Path = ":
            member = line.removeprefix("Path = ").strip()
            if member != str(rar):
                members.add(member)
else:
    members = set(subprocess.run(["tar", "-tf", str(rar)], capture_output=True, text=True, check=True).stdout.splitlines())
label_member = "Refuge/REFUGE-Test-GT/Glaucoma_label_and_Fovea_location.xlsx"
assert label_member in members


def read_rar(member):
    if shutil.which("7z"):
        output = Path("/tmp/refuge_rar_member")
        output.mkdir(parents=True, exist_ok=True)
        for old in output.iterdir():
            old.unlink()
        subprocess.run(["7z", "e", "-y", f"-o{output}", str(rar), member], capture_output=True, check=True)
        files = list(output.iterdir())
        if len(files) != 1:
            raise RuntimeError(f"Expected one extracted file for {member}, got {files}")
        return files[0].read_bytes()
    return subprocess.run(["tar", "-xOf", str(rar), member], capture_output=True, check=True).stdout


book = openpyxl.load_workbook(io.BytesIO(read_rar(label_member)), read_only=True, data_only=True)
rows = list(book.active.values)
assert rows[0][:3] == ("ID", "ImgName", "Label(Glaucoma=1)")
test_labels = {str(row[1]): int(row[2]) for row in rows[1:]}
assert len(test_labels) == 400 and set(test_labels.values()) == {0, 1}
assert set(test_labels) == {f"T{i:04d}.jpg" for i in range(1, 401)}
with (audit / "refuge_test_labels_official.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["image", "label"])
    writer.writerows(sorted(test_labels.items()))

manifest_path = audit / "refuge_manifest.csv"
manifest = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
assert len(manifest) == 1200
for record in manifest:
    if record["split"] == "test":
        record["label"] = test_labels[Path(record["image"]).name]
with manifest_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)

# Check every expected original file name, then compare representative image bytes
# and mask pixels across train, validation, and test.
checks = []
with zipfile.ZipFile(combined) as z:
    combined_members = set(z.namelist())
    for split in ("train", "val", "test"):
        subset = sorted((r for r in manifest if r["split"] == split), key=lambda r: r["image"])
        assert len(subset) == 400
        for record in subset:
            filename = Path(record["image"]).name
            stem = Path(filename).stem
            label = int(record["label"])
            if split == "train":
                cls = "Glaucoma" if label else "Non-Glaucoma"
                original_image = f"Refuge/Training400/{cls}/{filename}"
                original_mask = f"Refuge/Annotation-Training400/Disc_Cup_Masks/{cls}/{stem}.bmp"
            elif split == "val":
                original_image = f"Refuge/REFUGE-Validation400/REFUGE-Validation400/{filename}"
                original_mask = f"Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks/{stem}.bmp"
            else:
                original_image = f"Refuge/Test400/{filename}"
                original_mask = f"Refuge/REFUGE-Test-GT/Disc_Cup_Masks/{'G' if label else 'N'}/{stem}.bmp"
            assert original_image in members and original_mask in members, (original_image, original_mask)
            assert record["image"] in combined_members and record["mask"] in combined_members
        for record in (subset[0], subset[199], subset[-1]):
            filename = Path(record["image"]).name
            stem = Path(filename).stem
            label = int(record["label"])
            if split == "train":
                cls = "Glaucoma" if label else "Non-Glaucoma"
                original_image = f"Refuge/Training400/{cls}/{filename}"
                original_mask = f"Refuge/Annotation-Training400/Disc_Cup_Masks/{cls}/{stem}.bmp"
            elif split == "val":
                original_image = f"Refuge/REFUGE-Validation400/REFUGE-Validation400/{filename}"
                original_mask = f"Refuge/REFUGE-Validation400-GT/REFUGE-Validation400-GT/Disc_Cup_Masks/{stem}.bmp"
            else:
                original_image = f"Refuge/Test400/{filename}"
                original_mask = f"Refuge/REFUGE-Test-GT/Disc_Cup_Masks/{'G' if label else 'N'}/{stem}.bmp"
            assert read_rar(original_image) == z.read(record["image"]), (split, filename, "image differs")
            original = np.asarray(Image.open(io.BytesIO(read_rar(original_mask))))
            repackaged = np.asarray(Image.open(io.BytesIO(z.read(record["mask"]))))
            assert original.shape == repackaged.shape
            assert np.array_equal(np.where(original == 255, 0, np.where(original == 128, 1, 2)), repackaged), (split, filename, "mask differs")
            checks.append({"split": split, "image": filename, "image_bytes_identical": True, "mask_pixels_equivalent": True})

summary = {
    "source": "Figshare article 26049574, Refuge.rar, published MD5 verified separately",
    "original_members": len(members),
    "official_test_labels": dict(Counter(test_labels.values())),
    "repackaged_audit": "All 1200 original image/mask names found; 9 sampled images byte-identical and masks pixel-equivalent after value mapping",
    "sample_checks": checks,
}
(audit / "refuge_original_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "sample_checks"}, indent=2))
