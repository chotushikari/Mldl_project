"""Print metadata for candidate original dataset sources without downloading files."""

from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen

sys.stdout.reconfigure(encoding="utf-8")


ARTICLES = {
    "ACRIMA": "https://api.figshare.com/v2/articles/7613135",
    "ORIGA candidate": "https://api.figshare.com/v2/articles/24549217",
    "REFUGE candidate": "https://api.figshare.com/v2/articles/26049574",
}


def get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "MLDL-dataset-audit/0.1"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


for name, url in ARTICLES.items():
    try:
        article = get_json(url)
        print(f"{name}: {article.get('title')} | doi={article.get('doi')} | license={article.get('license')}")
        for item in article.get("files", []):
            print(f"  {item.get('name')} | {item.get('size')} bytes | {item.get('download_url')}")
    except Exception as exc:
        print(f"{name}: ERROR {type(exc).__name__}: {exc}")

url = "https://medimrg.webs.ull.es/RIM-ONE-r3.zip"
try:
    request = Request(url, method="HEAD", headers={"User-Agent": "MLDL-dataset-audit/0.1"})
    with urlopen(request, timeout=30) as response:
        print(f"RIM-ONE r3: {response.status} | {response.headers.get('Content-Length')} bytes | {url}")
except Exception as exc:
    print(f"RIM-ONE r3: ERROR {type(exc).__name__}: {exc}")
