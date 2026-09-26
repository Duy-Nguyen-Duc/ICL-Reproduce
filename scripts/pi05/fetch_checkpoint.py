"""Parallel ranged download of a public openpi checkpoint, verified against the bucket's MD5
(or CRC32C for composite objects, which carry no MD5; needs google-crc32c).

openpi's own downloader streams one file at a time (~1-3 MB/s here); this splits every
object into byte ranges fetched concurrently, then checks each file against the bucket's
MD5 before moving the tree into place.
"""
import argparse
import base64
import hashlib
import json
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "openpi-assets"
CHUNK = 64 * 2**20

parser = argparse.ArgumentParser()
parser.add_argument("--prefix", default="checkpoints/pi05_libero")
parser.add_argument("--dest", default="/dev/shm/openpi_cache/openpi-assets")
parser.add_argument("--workers", type=int, default=32)
args = parser.parse_args()

listing = json.load(urllib.request.urlopen(
    f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix={args.prefix}/"
    "&fields=items(name,size,md5Hash,crc32c)"))["items"]
staging = os.path.join(args.dest, args.prefix + ".fetching")
final = os.path.join(args.dest, args.prefix)
jobs = []
for item in listing:
    path = os.path.join(staging, item["name"][len(args.prefix) + 1:])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = int(item["size"])
    with open(path, "wb") as fh:
        fh.truncate(size)
    jobs += [(item["name"], path, start, min(start + CHUNK, size) - 1)
             for start in range(0, max(size, 1), CHUNK) if size]


def fetch(job):
    name, path, start, end = job
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                f"https://storage.googleapis.com/{BUCKET}/{name}",
                headers={"Range": f"bytes={start}-{end}"})
            data = urllib.request.urlopen(req, timeout=120).read()
            assert len(data) == end - start + 1
            with open(path, "r+b") as fh:
                fh.seek(start)
                fh.write(data)
            return len(data)
        except Exception:
            if attempt == 4:
                raise


done = 0
total = sum(int(i["size"]) for i in listing)
with ThreadPoolExecutor(args.workers) as pool:
    for n in pool.map(fetch, jobs):
        done += n
        if done % (1 << 30) < n:
            print(f"{done / 1e9:.1f}/{total / 1e9:.1f} GB", flush=True)

for item in listing:
    path = os.path.join(staging, item["name"][len(args.prefix) + 1:])
    if "md5Hash" in item:
        digest, expected = hashlib.md5(), item["md5Hash"]
    else:
        import google_crc32c
        digest, expected = google_crc32c.Checksum(), item["crc32c"]
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    if base64.b64encode(digest.digest()).decode() != expected:
        raise SystemExit(f"checksum mismatch: {item['name']}")
if os.path.exists(final):
    shutil.rmtree(final)
os.rename(staging, final)
print(f"DONE {final} ({len(listing)} files, {total / 1e9:.2f} GB, MD5 verified)")
