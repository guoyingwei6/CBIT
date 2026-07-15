#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import shutil
import struct
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source_dir: Path, output_dir: Path) -> None:
    manifest_path = source_dir / "gbc-reference.json"
    manifest = json.loads(manifest_path.read_text())
    compressed = source_dir / manifest["file"]
    if compressed.stat().st_size != manifest["compressedBytes"]:
        raise ValueError("Compressed GBC reference size mismatch")
    if sha256(compressed) != manifest["compressedSha256"]:
        raise ValueError("Compressed GBC reference checksum mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = output_dir / "reference.bin"
    digest = hashlib.sha256()
    size = 0
    try:
        with gzip.open(compressed, "rb") as source, payload.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                target.write(chunk)
        if size != manifest["uncompressedBytes"]:
            raise ValueError("GBC reference payload size mismatch")
        if digest.hexdigest() != manifest["payloadSha256"]:
            raise ValueError("GBC reference payload checksum mismatch")
        with payload.open("rb") as source:
            magic, schema, snps, breeds, reserved = struct.unpack(
                "<8sIIII", source.read(manifest["headerBytes"])
            )
        if (
            magic != b"CBITGBC1"
            or schema != manifest["schemaVersion"]
            or snps != manifest["snps"]
            or breeds != len(manifest["breeds"])
            or reserved != 0
        ):
            raise ValueError("GBC reference header mismatch")
        shutil.copy2(manifest_path, output_dir / "reference.json")
    except Exception:
        payload.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and unpack the shared browser and Cloud Run GBC reference."
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    prepare(args.source_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()

