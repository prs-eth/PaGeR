"""Shared I/O helpers used by multiple PaGeR dataloaders."""

from __future__ import annotations

from pathlib import Path
from struct import unpack

import numpy as np

_TAG_FLOAT = 202021.25


def read_dpt(path) -> np.ndarray:
    """Read a 360monodepth .dpt file (magic, int32 w, int32 h, H*W float32) → (H, W) float32."""
    path = Path(path)
    assert path.suffix == ".dpt", f"expected .dpt, got {path.suffix}"
    with open(path, "rb") as fid:
        tag = unpack("f", fid.read(4))[0]
        width = unpack("i", fid.read(4))[0]
        height = unpack("i", fid.read(4))[0]
        assert tag == _TAG_FLOAT, f"{path}: wrong magic tag (endianness?)"
        return np.fromfile(fid, np.float32).reshape(height, width)
