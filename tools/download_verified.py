"""Download only source-verified dataset archives, with resumable transfers."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SOURCES = {
    "acrima": ("https://ndownloader.figshare.com/files/14137700", 23608145, "ACRIMA_Database.zip"),
    "rim_one_r3": ("https://medimrg.webs.ull.es/RIM-ONE-r3.zip", 203123809, "RIM-ONE-r3.zip"),
    "refuge_candidate": ("https://ndownloader.figshare.com/files/47095942", 4265372818, "REFUGE_figshare.rar"),
    "origa_candidate": ("https://ndownloader.figshare.com/files/43119880", 5956275477, "ORIGA_candidate_figshare.zip"),
}


def download(name: str, root: Path) -> None:
    url, size, filename = SOURCES[name]
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    partial = root / f"{filename}.part"
    if target.exists() and target.stat().st_size == size:
        print(f"{name}: already downloaded", flush=True)
        return
    for attempt in range(1, 8):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            request = Request(url, headers={"User-Agent": "MLDL-research/0.1", **({"Range": f"bytes={offset}-"} if offset else {})})
            with urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            break
        except (ConnectionResetError, TimeoutError, URLError, OSError) as exc:
            if attempt == 7:
                raise
            print(f"{name}: transfer interrupted ({type(exc).__name__}); retrying from byte {partial.stat().st_size if partial.exists() else 0}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    actual = partial.stat().st_size
    if actual != size:
        raise RuntimeError(f"{name}: expected {size} bytes, got {actual}; partial retained")
    partial.replace(target)
    sha = hashlib.sha256()
    with target.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            sha.update(chunk)
    (root / f"{filename}.provenance.json").write_text(json.dumps({"dataset": name, "source_url": url, "bytes": size, "sha256": sha.hexdigest()}, indent=2))
    print(f"{name}: {target} | sha256={sha.hexdigest()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    download(args.dataset, args.output)
