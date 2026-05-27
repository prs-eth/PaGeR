"""Structured3D: synthetic indoor renderings with GT depth + normals.

Normals are remapped (n_x, n_y, n_z) → (-n_z, n_y, -n_x) to match the camera-centered ERP frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dataloaders._base import PanoDataset


class Structured3D(PanoDataset):
    HEIGHT, WIDTH = 512, 1024
    SUBDIR = "Structured3D"

    def _scan(self):
        broken: set[Path] = set()
        broken_list = self.data_path / "broken_samples.txt"
        if broken_list.exists():
            for line in broken_list.read_text().splitlines():
                entry = line.split(" [")[0].strip()
                if entry:
                    broken.add(Path(entry))

        samples = []
        for sid in range(3250, 3500):
            scene_dir = self.data_path / f"scene_{sid:05d}" / "2D_rendering"
            if not scene_dir.is_dir():
                continue
            for rgb in scene_dir.glob("*/panorama/full/rgb_rawlight.png"):
                depth = rgb.parent / "depth.png"
                normals = rgb.parent / "normal.png"
                if not (depth.exists() and normals.exists()):
                    continue
                if rgb in broken or depth in broken or normals in broken:
                    continue
                samples.append({
                    "id": f"{rgb.parts[-6]}_{rgb.parts[-4]}",
                    "rgb": rgb,
                    "depth": depth,
                    "normals": normals,
                })
        return samples

    def _load_depth(self, entry):
        return np.array(Image.open(entry["depth"])) / 1000.0  # mm → m

    def _load_normals(self, entry):
        # PNG [0, 255] → [-1, 1], then remap to camera-centered ERP.
        n = np.array(Image.open(entry["normals"])).astype(np.float32) / 128.0 - 1.0
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
        return np.stack([-nz, ny, -nx], axis=-1)
