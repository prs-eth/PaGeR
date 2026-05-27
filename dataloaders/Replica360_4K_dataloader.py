"""Replica360_4K: synthetic indoor renderings (RGB + .dpt depth + scene mask)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from dataloaders._base import PanoDataset
from dataloaders._io import read_dpt


class Replica360_4K(PanoDataset):
    HEIGHT, WIDTH = 2048, 4096
    SUBDIR = "Replica360_4K"

    def _scan(self):
        samples = []
        for scene_dir in sorted(p for p in self.data_path.iterdir() if p.is_dir()):
            for rgb in sorted(scene_dir.glob("*_rgb_pano.jpg")):
                samples.append({
                    "id": f"{scene_dir.name}_{rgb.stem[:4]}",
                    "rgb": rgb,
                    "depth": rgb.with_name(rgb.name.replace("_rgb_pano.jpg", "_depth_pano.dpt")),
                    "mask": rgb.with_name(rgb.name.replace("_rgb_pano.jpg", "_mask_pano.png")),
                })
        return samples

    def _load_depth(self, entry):
        return read_dpt(entry["depth"])

    def _scene_mask(self, entry, _depth):
        return np.array(Image.open(entry["mask"]).convert("L")) > 0
