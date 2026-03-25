#!/usr/bin/env python3
"""
download_models.py — Download pre-trained ONNX models for demos and benchmarks.

Available models:
  resnet50   — ResNet-50 image classification (vision/classification/resnet)
  identity   — Tiny identity model (generated locally)

Usage:
  python3 scripts/download_models.py resnet50
  python3 scripts/download_models.py resnet50 identity
"""

import os
import sys
import urllib.request
import hashlib

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "resnet50": {
        "url": (
            "https://github.com/onnx/models/raw/main/validated/"
            "vision/classification/resnet/model/resnet50-v1-12.onnx"
        ),
        "filename": "resnet50.onnx",
        "sha256": None,  # Checksum optional; set to verify download integrity
    },
    "identity": {
        "url": None,  # Generated locally
        "filename": "test_identity.onnx",
        "generator": "scripts/generate_test_models.py",
    },
}


def download(model_name: str) -> None:
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        return

    info = MODELS[model_name]

    if info.get("generator"):
        # Generate locally
        gen_script = os.path.join(os.path.dirname(__file__), "..", info["generator"])
        os.system(f"{sys.executable} {gen_script}")
        return

    url = info["url"]
    dest = os.path.join(OUTPUT_DIR, info["filename"])

    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return

    print(f"Downloading {model_name} from {url} ...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as e:
        print(f"\nERROR downloading {model_name}: {e}")
        return

    print(f"\nSaved to: {dest}")

    if info.get("sha256"):
        _verify_checksum(dest, info["sha256"])


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        print(f"\r  {pct:.1f}%  ({downloaded // 1024} KB / {total_size // 1024} KB)",
              end="", flush=True)


def _verify_checksum(path: str, expected: str) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"WARNING: SHA-256 mismatch for {path}")
        print(f"  expected: {expected}")
        print(f"  actual:   {actual}")
    else:
        print(f"  SHA-256 OK")


if __name__ == "__main__":
    models = sys.argv[1:] if len(sys.argv) > 1 else ["resnet50"]
    for m in models:
        download(m)
