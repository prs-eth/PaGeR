"""Stanford 2D-3D-S: 4K indoor panoramas, 16-bit PNG depth (scale 128/65535 → m); 15% pole crop on mask only."""

from __future__ import annotations

import numpy as np
from PIL import Image

from dataloaders._base import PanoDataset


class Stanford2D3DS(PanoDataset):
    HEIGHT, WIDTH = 2048, 4096
    SUBDIR = "Stanford2D3DS"
    POLE_CROP_FRAC = 0.15

    def _scan(self):
        return [
            {
                "id": p.name[:-37],
                "rgb": p,
                "depth": p.parent.parent / "depth" / p.name.replace("_rgb.png", "_depth.png"),
            }
            for p in sorted(self.data_path.rglob("*_rgb.png"))
        ]

    def _load_depth(self, entry):
        return np.array(Image.open(entry["depth"])) * (128.0 / 65535.0)
