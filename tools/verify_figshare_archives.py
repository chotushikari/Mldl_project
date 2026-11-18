"""Compare completed candidate archives with Figshare's published MD5 hashes."""
import hashlib
from pathlib import Path


SOURCES = {
    "REFUGE": (Path("data/raw/REFUGE_figshare.rar"), "963df1526e63c4e437dba37dcd8c89ae"),
    "ORIGA candidate": (Path("data/raw/ORIGA_candidate_figshare.zip"), "8838f860d6d7de42ac00034e3b1ca2df"),
}
for name, (path, expected) in SOURCES.items():
    if not path.exists():
        print(name, "not complete")
        continue
    digest = hashlib.md5()  # Source integrity check only; not a security signature.
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    print(name, actual, "matches Figshare" if actual == expected else "MISMATCH")
    assert actual == expected, f"{name} archive checksum differs from Figshare metadata"
