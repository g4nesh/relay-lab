"""Resume the pinned public X-VLA weights over bounded, verified HTTP ranges.

Normally huggingface_hub.snapshot_download is preferable. This helper is for a
connection that repeatedly times out during the 3.52 GB contiguous transfer.
Metadata/tokenizer assets still come from Hugging Face's regular downloader.
Nothing is loaded until the full publisher SHA-256 has been verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from relay_lab.cloth.pretrained import (
    BART_ID,
    BART_REVISION,
    XVLA_ID,
    XVLA_REVISION,
    XVLA_WEIGHT_BYTES,
    XVLA_WEIGHT_SHA256,
)

WEIGHT_BYTES = XVLA_WEIGHT_BYTES
WEIGHT_SHA256 = XVLA_WEIGHT_SHA256
CHUNK_BYTES = 4 * 1024 * 1024


def download(cache: Path, workers: int = 8):
    from huggingface_hub import snapshot_download

    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        XVLA_ID, revision=XVLA_REVISION, cache_dir=cache, allow_patterns=["*.json", "README.md"]
    )
    snapshot_download(
        BART_ID,
        revision=BART_REVISION,
        cache_dir=cache,
        allow_patterns=[
            "config.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    )
    base = cache / "models--lerobot--xvla-folding"
    destination = base / "blobs" / WEIGHT_SHA256
    if destination.is_file() and _sha(destination) == WEIGHT_SHA256:
        _link_checkpoint(base)
        return destination
    parts = base / "range-parts"
    parts.mkdir(exist_ok=True)
    count = (WEIGHT_BYTES + CHUNK_BYTES - 1) // CHUNK_BYTES
    url = (
        f"https://huggingface.co/{XVLA_ID}/resolve/{XVLA_REVISION}/model.safetensors?download=true"
    )

    def fetch(index):
        start = index * CHUNK_BYTES
        end = min(WEIGHT_BYTES - 1, start + CHUNK_BYTES - 1)
        path = parts / f"{index:05}.part"
        if path.is_file() and path.stat().st_size == end - start + 1:
            return path.stat().st_size
        for attempt in range(4):
            try:
                request = Request(
                    url + f"&range_part={index}", headers={"Range": f"bytes={start}-{end}"}
                )
                with urlopen(request, timeout=60) as response:
                    if (
                        response.status != 206
                        or response.headers.get("content-range")
                        != f"bytes {start}-{end}/{WEIGHT_BYTES}"
                    ):
                        raise RuntimeError(
                            "Model host did not honor the exact requested byte range"
                        )
                    temporary = path.with_suffix(".tmp")
                    received = 0
                    with temporary.open("wb") as stream:
                        while block := response.read(256 * 1024):
                            stream.write(block)
                            received += len(block)
                    if received != end - start + 1:
                        raise RuntimeError("Truncated model range")
                temporary.replace(path)
                return received
            except (OSError, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        raise AssertionError("unreachable")

    started = time.monotonic()
    received = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count)]
        for completed, future in enumerate(as_completed(futures), start=1):
            received += future.result()
            if completed % 16 == 0 or completed == count:
                print(
                    json.dumps(
                        {
                            "parts": completed,
                            "total_parts": count,
                            "received_mb": round(received / 1e6),
                            "seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
    temporary = destination.with_suffix(".verified-download")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(count):
            with (parts / f"{index:05}.part").open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        temporary.unlink()
        raise RuntimeError("Full checkpoint SHA-256 mismatch; weights were not installed")
    temporary.replace(destination)
    _link_checkpoint(base)
    print(
        json.dumps({"complete": True, "sha256": WEIGHT_SHA256, "bytes": WEIGHT_BYTES}), flush=True
    )
    return destination


def _link_checkpoint(base):
    """Recover the snapshot link if a previous run stopped after verification."""
    link = base / "snapshots" / XVLA_REVISION / "model.safetensors"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() != (base / "blobs" / WEIGHT_SHA256).resolve():
            raise RuntimeError("Pinned model snapshot has an unexpected checkpoint path")
        return
    link.symlink_to("../../blobs/" + WEIGHT_SHA256)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(".cache/huggingface/hub"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    download(args.cache, args.workers)
