"""Download the user-selected DRISHTI-GS Kaggle mirror with explicit provenance."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASET = "lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
OUT = Path("data/provisional/drishti_gs1")
TARGET = OUT / "drishti_gs1_kaggle_mirror.zip"
PARTIAL = OUT / "drishti_gs1_kaggle_mirror.zip.part"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
if TARGET.exists():
    print(json.dumps({"state": "complete", "path": str(TARGET), "bytes": TARGET.stat().st_size, "sha256": sha256(TARGET)}, indent=2))
    raise SystemExit

for attempt in range(1, 8):
    offset = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    try:
        headers = {"User-Agent": "MLDL-research/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(URL, headers=headers)
        with urlopen(request, timeout=180) as response:
            if offset and response.status != 206:
                offset = 0
                PARTIAL.unlink(missing_ok=True)
            with PARTIAL.open("ab" if offset else "wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        break
    except (HTTPError, URLError, TimeoutError, ConnectionResetError, OSError) as exc:
        if attempt == 7:
            raise
        print(f"Transfer interrupted ({type(exc).__name__}); retained {PARTIAL.stat().st_size if PARTIAL.exists() else 0} bytes", flush=True)
        time.sleep(min(2 ** attempt, 30))

PARTIAL.replace(TARGET)
record = {
    "dataset": "DRISHTI-GS1",
    "source": "Third-party Kaggle mirror",
    "uploader": "lokeshsaipureddi",
    "url": f"https://www.kaggle.com/datasets/{DATASET}",
    "kaggle_metadata_license": "Kaggle page advertises CC0",
    "official_terms": "Official IIIT page limits use to academic/research and prohibits redistribution except within the registered team",
    "mirror_based_experiment": True,
    "official_source_verified": False,
    "bytes": TARGET.stat().st_size,
    "sha256": sha256(TARGET),
}
(OUT / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2))
