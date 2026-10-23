"""Restore manifest-listed EPFL RAW/scan pairs, verifying every SHA256.

python -I tools/eval/download_epfl.py --data /workspace/filmlab-data
External data stays outside the checkout. No downloaded code is executed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from pathlib import Path
import re
import urllib.request


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(task):
    url, path, expected = task
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"Existing file has wrong SHA256; preserved: {path}")
        return f"verified existing {path.name}"
    part = path.with_suffix(path.suffix + ".part")
    if part.exists():
        raise ValueError(f"Partial file exists; inspect before retry: {part}")
    with urllib.request.urlopen(url, timeout=90) as response, part.open("xb") as stream:
        while block := response.read(1 << 20):
            stream.write(block)
    if sha256(part) != expected:
        raise ValueError(f"Downloaded SHA256 mismatch; not accepted: {part}")
    part.rename(path)
    return f"downloaded and verified {path.name} ({path.stat().st_size} bytes)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scenes", nargs="*")
    args = ap.parse_args()
    if not 1 <= args.jobs <= 8:
        ap.error("--jobs must be between 1 and 8")
    manifest = Path(__file__).resolve().with_name("epfl_800t_manifest.csv")
    rows = list(csv.DictReader(manifest.open()))
    selected = set(args.scenes) if args.scenes else {r["scene_id"] for r in rows}
    unknown = selected - {r["scene_id"] for r in rows}
    if unknown:
        ap.error(f"Unknown scenes: {sorted(unknown)}")
    root = (args.data / "epfl" / "data").resolve()
    tasks = []
    for r in rows:
        if r["scene_id"] not in selected:
            continue
        urls = [u.strip() for u in r["download_url"].split("|")]
        if len(urls) != 2:
            raise ValueError(f"Expected film/raw URLs for {r['scene_id']}")
        for kind, source in zip(("film", "raw"), urls):
            match = re.fullmatch(r"https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)", source)
            digest = r[kind + "_sha256"]
            path = (root / r[kind + "_file"]).resolve()
            if not match or not re.fullmatch(r"[0-9a-f]{64}", digest) or not path.is_relative_to(root):
                raise ValueError(f"Invalid manifest record: {r['scene_id']} / {kind}")
            url = f"https://drive.usercontent.google.com/download?id={match[1]}&export=download&confirm=t"
            tasks.append((url, path, digest))
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(download, task): task[1] for task in tasks}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                failed.append(str(futures[future]))
                print(f"FAILED {futures[future]}: {exc}", flush=True)
    print(f"Verified {len(tasks) - len(failed)}/{len(tasks)} files", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
