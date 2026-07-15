import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


KEY_MULTIPLIER = 1_000_000_000


@dataclass(frozen=True)
class ReferenceData:
    keys: NDArray[np.uint64]
    values: NDArray[np.float64]
    breeds: tuple[str, ...]


def load_reference(asset_dir: Path) -> ReferenceData:
    manifest = json.loads((asset_dir / "reference.json").read_text())
    payload = asset_dir / "reference.bin"
    breeds = tuple(manifest["breeds"])
    snps = int(manifest["snps"])
    expected_size = int(manifest["valuesOffset"]) + snps * len(breeds) * 8
    if payload.stat().st_size != expected_size:
        raise ValueError("Invalid GBC reference payload size")

    with payload.open("rb") as source:
        magic, schema, header_snps, header_breeds, reserved = struct.unpack(
            "<8sIIII", source.read(int(manifest["headerBytes"]))
        )
    if (
        magic != b"CBITGBC1"
        or schema != manifest["schemaVersion"]
        or header_snps != snps
        or header_breeds != len(breeds)
        or reserved != 0
    ):
        raise ValueError("Invalid GBC reference header")

    key_pairs = np.memmap(
        payload,
        mode="r",
        dtype="<u4",
        offset=int(manifest["keysOffset"]),
        shape=(snps, 2),
    )
    keys = key_pairs[:, 0].astype(np.uint64)
    keys *= KEY_MULTIPLIER
    keys += key_pairs[:, 1]
    keys.setflags(write=False)
    values = np.memmap(
        payload,
        mode="r",
        dtype="<f8",
        offset=int(manifest["valuesOffset"]),
        shape=(snps, len(breeds)),
    )
    if np.any(keys[1:] <= keys[:-1]):
        raise ValueError("GBC reference keys are not strictly sorted")
    return ReferenceData(keys=keys, values=values, breeds=breeds)

