from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("yt-dlp"):
        subprocess.check_call(["yt-dlp", "-o", str(dest), url])
    else:
        urllib.request.urlretrieve(url, dest)  # nosec - user-provided challenge video links


def main() -> int:
    parser = argparse.ArgumentParser(description="Download challenge clips from a JSON manifest.")
    parser.add_argument("manifest", help="JSON with videos[].url and videos[].path")
    args = parser.parse_args()
    data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for item in data.get("videos", []):
        if "url" not in item:
            continue
        download(item["url"], Path(item["path"]))
        print(f"downloaded {item['url']} -> {item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
