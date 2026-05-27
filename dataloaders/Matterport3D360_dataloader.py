"""Matterport3D360: real-scanner indoor panoramas with .dpt depth; 14% pole crop on RGB and mask."""

from __future__ import annotations

from dataloaders._base import PanoDataset
from dataloaders._io import read_dpt


class Matterport3D360(PanoDataset):
    HEIGHT, WIDTH = 1024, 2048
    SUBDIR = "Matterport3D360"
    POLE_CROP_FRAC = 0.14
    POLE_CROP_RGB = True

    def _scan(self):
        scenes = set((self.data_path / "scenes_test.txt").read_text().split())
        samples = []
        for partition in self.data_path.glob("*/data/*"):
            if not partition.is_dir() or partition.name not in scenes:
                continue
            for rgb in sorted(partition.glob("*_rgb.png")):
                key = rgb.name[: -len("_rgb.png")]
                depth = rgb.with_name(f"{key}_depth.dpt")
                if depth.exists():
                    samples.append({"id": key, "rgb": rgb, "depth": depth})
        return samples

    def _load_depth(self, entry):
        return read_dpt(entry["depth"])
