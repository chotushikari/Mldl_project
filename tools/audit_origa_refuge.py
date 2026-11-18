"""Audit ORIGA and REFUGE members in the checksum-verified combined archive."""
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


archive_path = Path("data/raw/ORIGA_candidate_figshare.zip")
audit_root = Path("data/audit")
audit_root.mkdir(parents=True, exist_ok=True)


def sample_panel(z, image_member, mask_member, destination):
    image = Image.open(io.BytesIO(z.read(image_member))).convert("RGB")
    mask = Image.open(io.BytesIO(z.read(mask_member))).convert("L")
    assert image.size == mask.size, (image_member, mask_member, image.size, mask.size)
    image = image.resize((512, 512))
    mask = mask.resize((512, 512), Image.Resampling.NEAREST)
    values = np.asarray(mask)
    assert set(np.unique(values)) == {0, 1, 2}
    original = np.asarray(image).copy()
    overlay = original.copy()
    overlay[values == 1] = 0.55 * overlay[values == 1] + 0.45 * np.array([0, 220, 110])
    overlay[values == 2] = 0.50 * overlay[values == 2] + 0.50 * np.array([255, 180, 0])
    panel = Image.new("RGB", (1024, 540), "white")
    panel.paste(image, (0, 0))
    panel.paste(Image.fromarray(overlay.astype("uint8")), (512, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 518), "Original", fill="black")
    draw.text((522, 518), "Mask values: 1=outer ring, 2=inner region", fill="black")
    panel.save(destination)


with zipfile.ZipFile(archive_path) as z:
    members = set(z.namelist())
    origa = list(csv.DictReader(io.TextIOWrapper(z.open("ORIGA/OrigaList.csv"), encoding="utf-8-sig")))
    assert len(origa) == 650
    assert len({r["Filename"] for r in origa}) == len(origa)
    assert {r["Glaucoma"] for r in origa} == {"0", "1"}
    assert {r["Set"] for r in origa} == {"A", "B"}
    origa_manifest = []
    for r in origa:
        image = "ORIGA/Images/" + r["Filename"]
        mask = "ORIGA/Masks/" + Path(r["Filename"]).with_suffix(".png").name
        assert image in members and mask in members, (image, mask)
        origa_manifest.append({"image": image, "mask": mask, "label": r["Glaucoma"], "provided_set": r["Set"], "eye": r["Eye"], "expert_cdr": r["ExpCDR"]})
    with (audit_root / "origa_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(origa_manifest[0]))
        writer.writeheader()
        writer.writerows(origa_manifest)

    refuge_manifest = []
    refuge_summary = {}
    for split in ("train", "val", "test"):
        index = json.load(z.open(f"REFUGE/{split}/index.json"))
        records = list(index.values())
        assert len(records) == 400
        assert len({r["ImgName"] for r in records}) == len(records)
        labels = Counter(str(r.get("Label", "missing")) for r in records)
        if split == "test":
            assert labels == {"missing": 400}, "Do not infer missing test labels"
        else:
            assert set(labels) == {"0", "1"}
        for r in records:
            stem = Path(r["ImgName"]).stem
            image = f"REFUGE/{split}/Images/{r['ImgName']}"
            mask = f"REFUGE/{split}/Masks/{stem}.png"
            assert image in members and mask in members, (image, mask)
            refuge_manifest.append({"image": image, "mask": mask, "split": split, "label": r.get("Label", "")})
        refuge_summary[split] = {"images": len(records), "labels": dict(labels)}
    with (audit_root / "refuge_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(refuge_manifest[0]))
        writer.writeheader()
        writer.writerows(refuge_manifest)

    sample_panel(z, origa_manifest[0]["image"], origa_manifest[0]["mask"], audit_root / "origa_sample.png")
    sample_panel(z, refuge_manifest[0]["image"], refuge_manifest[0]["mask"], audit_root / "refuge_sample.png")

summary = {
    "archive": str(archive_path),
    "source": "Figshare article 24549217, third-party combined archive; MD5 checked separately",
    "origa": {"images": len(origa_manifest), "labels": dict(Counter(r["label"] for r in origa_manifest)), "provided_sets": dict(Counter(r["provided_set"] for r in origa_manifest))},
    "refuge": refuge_summary,
    "mask_encoding_sample": "0 background, 1 outer ring, 2 inner region; verify semantics against source before final training",
}
(audit_root / "origa_refuge_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
